import joblib

model = joblib.load('model/spam_model.pkl')

print("=" * 60)
print("SPAM EMAIL DETECTOR")
print("=" * 60)
print("Enter an email message to check if it's spam or not.")
print("(Press Ctrl+C to exit)\n")

email = input("Enter email message: ")

result = model.predict([email])

if result[0] == 1:
    print("\n🚫 SPAM EMAIL")
else:
    print("\n✅ NOT SPAM EMAIL")

