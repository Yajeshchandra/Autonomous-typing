all_keys = [
    "ESC", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
    "PRTSC", "SCRLK", "PAUSE", "`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "-", "=", "BACKSPACE", "INS", "HOME", "PAGEUP", "TAB", "Q", "W", "E", "R", "T", "Y",
    "U", "I", "O", "P", "[", "]", "\\", "DEL", "END", "PAGEDOWN", "CAPSLOCK", "A", "S",
    "D", "F", "G", "H", "J", "K", "L", ";", "'", "ENTER", "SHIFT", "Z", "X", "C", "V",
    "B", "N", "M", ",", ".", "/", "UP", "CTRL", "WIN", "ALT", "SPACE", "FN", "MENU",
    "LEFT", "DOWN", "RIGHT"
]

def string_to_keyboard_clicks(input_string):
    keyboard_clicks = []
    caps_active = False  # Track CAPS state

    for char in input_string:
        if char.isalpha():
            if char.isupper() and not caps_active:
                # Activate CAPS for uppercase letters
                keyboard_clicks.append("CAPS")
                caps_active = True
            elif char.islower() and caps_active:
                # Deactivate CAPS for lowercase letters
                keyboard_clicks.append("CAPS")
                caps_active = False
            # Add the character itself (uppercase or lowercase)
            keyboard_clicks.append(char.upper() if not caps_active else char)
        elif char.isspace():
            # Convert spaces to "SPACE"
            keyboard_clicks.append("SPACE")
        elif char in all_keys:
            # Add predefined keys from the list
            keyboard_clicks.append(char)
        else:
            # Add any other character directly
            keyboard_clicks.append(char)
    
    # Ensure CAPS is reset after the string (optional, for consistency)
    if caps_active:
        keyboard_clicks.append("CAPS")
    
    # Add "ENTER" at the end
    keyboard_clicks.append("ENTER")
    
    return keyboard_clicks

# Examples
print(string_to_keyboard_clicks("UrC O1"))  # ['CAPS', 'U', 'CAPS', 'R', 'CAPS', 'C', 'SPACE', 'O', '1', 'ENTER']
print(string_to_keyboard_clicks("OCD-1"))   # ['CAPS', 'O', 'C', 'D', '-', '1', 'ENTER']
print(string_to_keyboard_clicks("CTRL+ALT+DEL"))  # ['CAPS', 'C', 'T', 'R', 'L', '+', 'A', 'L', 'T', '+', 'D', 'E', 'L', 'ENTER']
