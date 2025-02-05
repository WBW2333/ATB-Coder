import requests
import json

url = "https://api.aiskt.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"sk-7MNQ6d4C7YuJDUi1898b39Ab869c40C58bCbE21d418d88D5"
}
data = {
    "model": "gpt-3.5-turbo",
    "stream": False,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    if False:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if chunk['choices'][0].get('delta', {}).get('content'):
                    print(chunk['choices'][0]['delta']['content'], end='')
    else:
        print(response.json()['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")