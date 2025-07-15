import json

with open("action_state.json", 'r', encoding='utf-8') as json_file:
    action_state = json.load(json_file)

action = action_state["Action"]

with open(f"{action}.json", 'w', encoding='utf-8') as json_file:
    json.dump(action_state, json_file)

with open(f"{action}.json", 'r', encoding='utf-8') as json_file:
    action_state = json.load(json_file)

print(action_state)