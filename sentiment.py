from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis") # Load a pretrained  sentiment-analysis AI model, this used for detect the + or - 
        
    def analyze(self, text):
        result = self.analyzer(text)[0] # take the input of user then load it to the AI model 

        label = result["label"]
        score = result["score"]

        return label, score

    def needs_escalation(self, label, score):
        return label == "NEGATIVE" and score > 0.90 # base on the score and the label of Pos or Neg that it detected (must be Negative and more than 90% confidence)

test = SentimentAnalyzer()

text = "I'm having a really strange and unsuitable feeling of the function of computer password system !"

label, score = test.analyze(text)

print("Sentiment:", label)
print("Score:", score)

if test.needs_escalation(label, score):
    print("Recommended escalation: Contact human advisor.")

