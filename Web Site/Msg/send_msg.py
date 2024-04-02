from twilio.rest import Client
import keys

msg = input("Enter the message: ")
client = Client(keys.account_sid, keys.auth_token)

message = client.messages.create(
    body=msg,
    from_=keys.twilio_number,
    to=keys.target_number
    
)

print(message.body)