import json as _json
import re

TC = 'tool_call'
FN = 'function='
PM = 'parameter='
CF = '/function'
CP = '/parameter'
CT = '/tool_call'

def _parse_xml_tool_calls(text):
    """Parse XML-style tool_call blocks from small model output."""
    calls = []
    pos = 0
    while True:
        start = text.find(TC, pos)
        if start == -1:
            break
        func_start = text.find(FN, start)
        if func_start == -1:
            break
        func_start += len(FN)
        func_end = text.find(">", func_start)
        if func_end == -1:
            break
        func_name = text[func_start:func_end].strip()
        params = {}
        ppos = func_end + 1
        while True:
            param_start = text.find(PM, ppos)
            if param_start == -1:
                break
            param_start += len(PM)
            param_end_name = text.find(">", param_start)
            if param_end_name == -1:
                break
            param_name = text[param_start:param_end_name].strip()
            param_val_start = param_end_name + 1
            param_val_end = text.find(CP, param_val_start)
            if param_val_end == -1:
                break
            raw_val = text[param_val_start:param_val_end].strip()
            try:
                val = _json.loads(raw_val)
            except (_json.JSONDecodeError, ValueError):
                val = raw_val
            params[param_name] = val
            ppos = param_val_end + len(CP)
        calls.append({"name": func_name, "arguments": params})
        end = text.find(CT, func_end)
        pos = end + len(CT) if end != -1 else len(text)
    return calls


def _parse_bracket_tool_calls(text):
    """Parse bracket-style tool calls like [function_name(args)] used by LFM2.5."""
    calls = []
    # Match patterns like: [record_stm(content="hello")] or [get_stm_window()]
    pattern = re.compile(r'\[(\w+)\(([^)]*)\)\]', re.DOTALL)
    for m in pattern.finditer(text):
        func_name = m.group(1)
        args_str = m.group(2).strip()
        args = {}
        if args_str:
            # Parse key=value pairs: content="hello", name="world"
            # Handle quoted values
            kv_pattern = re.compile(r'(\w+)="([^"]*)"')
            kv_pairs = kv_pattern.findall(args_str)
            if kv_pairs:
                for k, v in kv_pairs:
                    args[k] = v
            else:
                # Try bare values
                kv_pattern2 = re.compile(r'(\w+)=([^,\s]+)')
                kv_pairs2 = kv_pattern2.findall(args_str)
                for k, v in kv_pairs2:
                    try:
                        v = _json.loads(v)
                    except (_json.JSONDecodeError, ValueError):
                        pass
                    args[k] = v
        calls.append({"name": func_name, "arguments": args})
    return calls


def parse_tool_calls(text):
    """Parse tool calls from any format: native, XML, or bracket notation."""
    # Try XML first
    xml_calls = _parse_xml_tool_calls(text)
    if xml_calls:
        return xml_calls
    # Try bracket notation (LFM2.5 etc.)
    bracket_calls = _parse_bracket_tool_calls(text)
    if bracket_calls:
        return bracket_calls
    return []
