from django.shortcuts import render
import pickle
# Create your views here.

vectorizer = pickle.load(open('F:/jupyter/Spam Email/spam_detection/vectorizer.pkl', 'rb'))
model = pickle.load(open('F:/jupyter/Spam Email/spam_detection/model.pkl', 'rb'))

def check_spam(request):
    if request.method == 'POST':

        email_text = request.POST.get('email_text')

        transformed_text = vectorizer.transform([email_text])

        prediction = model.predict(transformed_text)[0]

        result = "Spam" if prediction == 1 else "Not Spam"

        return render(request, 'spam_classifier/result.html', {'result': result})
    
    return render(request, 'spam_classifier/input.html')