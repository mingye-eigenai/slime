#!/usr/bin/env python3
"""
Convert trajectory JSON files from /data/zhanlin_data_augmentation_opus/logs_opus/
into the SFT JSONL format matching apex_sft_opus_mix_filled_clean_v2_opus_rewritten_zhanlin_filtered.jsonl.

Conversion:
1. Build system prompt by injecting tool schemas from available_tools into <tools> XML
2. Parse <tool_call> XML from assistant content to extract tool name/arguments
3. Text before <tool_call> goes into reasoning_content
4. Generate tool call IDs and link tool result messages
5. Final assistant message (no tool call) gets content as-is, reasoning_content empty
6. Mark failed tool call steps with step_loss_mask=0
"""

import json
import os
import re
import string
import random
import sys
from glob import glob


SYSTEM_PREAMBLE = """You are an agent that completes tasks independently.
Use the tools provided to you to complete the task to the best of your ability.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schemas}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def generate_tool_call_id():
    """Generate a tool call ID like toolu_01XXXXX..."""
    chars = string.ascii_letters + string.digits
    suffix = ''.join(random.choices(chars, k=24))
    return f"toolu_01{suffix}"


def build_system_prompt(available_tools):
    """Build system prompt with tool schemas injected."""
    tool_lines = []
    for tool in available_tools:
        tool_lines.append(json.dumps(tool, ensure_ascii=False))
    tool_schemas = "\n".join(tool_lines)
    return SYSTEM_PREAMBLE.format(tool_schemas=tool_schemas)


def parse_tool_call_from_content(content):
    """
    Parse <tool_call>...</tool_call> from assistant content.
    Returns (reasoning_text, tool_name, tool_arguments) or (content, None, None) if no tool call.
    """
    if not content or '<tool_call>' not in content:
        return content, None, None

    # Split on <tool_call>
    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
    if not match:
        return content, None, None

    # Text before <tool_call> is reasoning
    tool_call_start = content.find('<tool_call>')
    reasoning = content[:tool_call_start].strip()

    # Parse the JSON inside <tool_call>
    tool_json_str = match.group(1).strip()
    try:
        tool_data = json.loads(tool_json_str)
        tool_name = tool_data.get('name', '')
        tool_args = tool_data.get('arguments', {})
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)
        return reasoning, tool_name, tool_args
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Warning: Failed to parse tool_call JSON: {e}")
        print(f"  Raw: {tool_json_str[:200]}")
        return content, None, None


def is_tool_error(tool_content):
    """Check if a tool result indicates an error."""
    if not tool_content:
        return False
    error_patterns = [
        'Internal error:',
        'Error:',
        'validation error',
        'Missing required argument',
    ]
    return any(p in str(tool_content) for p in error_patterns)


def convert_trajectory(traj_data):
    """Convert a single trajectory dict to SFT format."""
    messages = traj_data.get('messages', [])
    available_tools = traj_data.get('available_tools', [])

    if not messages:
        return None

    # Build system prompt
    system_prompt = build_system_prompt(available_tools)

    sft_messages = []

    # Process messages
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get('role')

        if role == 'system':
            # Replace with our constructed system prompt
            sft_messages.append({
                'role': 'system',
                'content': system_prompt
            })
            i += 1

        elif role == 'user':
            sft_messages.append({
                'role': 'user',
                'content': msg.get('content', '')
            })
            i += 1

        elif role == 'assistant':
            content = msg.get('content', '') or ''
            reasoning, tool_name, tool_args = parse_tool_call_from_content(content)

            if tool_name is not None:
                # Assistant message with tool call
                call_id = generate_tool_call_id()

                assistant_msg = {
                    'role': 'assistant',
                    'reasoning_content': reasoning,
                    'content': '',
                    'tool_calls': [{
                        'function': {
                            'name': tool_name,
                            'arguments': tool_args
                        },
                        'id': call_id
                    }]
                }

                # Check if the next message is a tool result with error
                tool_result_content = ''
                if i + 1 < len(messages) and messages[i + 1].get('role') == 'tool':
                    tool_result_content = messages[i + 1].get('content', '')

                if is_tool_error(tool_result_content):
                    assistant_msg['step_loss_mask'] = 0

                sft_messages.append(assistant_msg)

                # Next should be tool result
                if i + 1 < len(messages) and messages[i + 1].get('role') == 'tool':
                    tool_msg = {
                        'role': 'tool',
                        'content': messages[i + 1].get('content', ''),
                        'id': call_id
                    }
                    if is_tool_error(tool_result_content):
                        tool_msg['step_loss_mask'] = 0
                    sft_messages.append(tool_msg)
                    i += 2
                else:
                    i += 1

            else:
                # Final assistant message (no tool call)
                assistant_msg = {
                    'role': 'assistant',
                    'content': content,
                    'reasoning_content': '',
                    'tool_calls': []
                }
                sft_messages.append(assistant_msg)
                i += 1

        elif role == 'tool':
            # Orphan tool message (shouldn't happen normally)
            call_id = generate_tool_call_id()
            sft_messages.append({
                'role': 'tool',
                'content': msg.get('content', ''),
                'id': call_id
            })
            i += 1

        else:
            i += 1

    return {'messages': sft_messages}


def main():
    input_dir = '/data/zhanlin_data_augmentation_opus/logs_opus'
    output_file = '/data/zhanlin_data_augmentation_opus_sft.jsonl'

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    json_files = sorted(glob(os.path.join(input_dir, '*.json')))
    print(f"Found {len(json_files)} trajectory files in {input_dir}")

    converted = 0
    skipped = 0
    errors = 0

    with open(output_file, 'w') as out_f:
        for filepath in json_files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath) as f:
                    traj_data = json.load(f)

                # Skip trajectories with errors
                if traj_data.get('error'):
                    print(f"  Skipping {filename}: has error field")
                    skipped += 1
                    continue

                result = convert_trajectory(traj_data)
                if result is None:
                    print(f"  Skipping {filename}: no messages")
                    skipped += 1
                    continue

                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                converted += 1

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                errors += 1

    print(f"\nDone!")
    print(f"  Converted: {converted}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_file}")


if __name__ == '__main__':
    main()
