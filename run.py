import requests

interests = ['less than 18', '24', 'Work', 'Chic']
# ['less than 18', '$25 - $49', 'Casual/Everyday', 'nice']
print(requests.get('http://127.0.0.1:8000/connections', json = {'interests': interests}).json())
