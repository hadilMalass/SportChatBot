import os
import nltk
import ssl
import random
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK data
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and get stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "number",
        "patterns": ["how many sport we have", "what is the number of sport "],
        "responses": ["we have 20 sport:football\nBasketball\nswimming\nrugby\nboxing\nsoccer\ncycling\nvolleyball\ncricket\ntennis\nathletics\nbaseball\ngolf\nskiing\ngymnastics\nbadminton\nfencing\nTable tennis\nsurfing\nhockey"]
    },
     {
        "tag": "football",
        "patterns": ["What is football?", "Tell me about football", "How is football played?"],
        "responses": ["Football is a team sport played with a spherical ball between two teams of 11 players. It is widely considered to be the most popular sport in the world."]
    },
    {
        "tag": "football_rules",
        "patterns": ["What are the rules of football?", "Explain the rules of football", "How do you play football?"],
        "responses": ["Football is played with two teams of 11 players on a rectangular field. The objective is to score by getting the ball into the opposing goal. Players can use any part of their body except their hands and arms to move the ball, while the goalkeepers are allowed to use their hands within the penalty area."]
    },
    {
        "tag": "basketball",
        "patterns": ["What is basketball?", "Tell me about basketball", "How is basketball played?"],
        "responses": ["Basketball is a team sport in which two teams, most commonly of five players each, oppose one another on a rectangular court, competing with the primary objective of shooting a basketball through the opponent's hoop."]
    },{
        "tag": "basketball_rules",
        "patterns": ["What are the rules of basketball?", "Explain the rules of basketball", "How do you play basketball?"],
        "responses": ["Basketball is played between two teams of five players each on a rectangular court. The objective is to score points by shooting the ball through the opponent's hoop. A game is played in four quarters of 10 or 12 minutes. The team with the most points at the end of the game wins."]
    },
    {
        "tag": "cricket",
        "patterns": ["What is cricket?", "Tell me about cricket", "How is cricket played?"],
        "responses": ["Cricket is a bat-and-ball game played between two teams of eleven players on a field at the centre of which is a 22-yard pitch with a wicket at each end, each comprising two bails balanced on three stumps."]
    },{
        "tag": "cricket_rules",
        "patterns": ["What are the rules of cricket?", "Explain the rules of cricket", "How do you play cricket?"],
        "responses": ["Cricket is played with two teams of eleven players each. The objective is to score runs by hitting the ball and running between the wickets. The fielding team tries to get the batsmen out by hitting the stumps or catching the ball. Each team takes turns batting and fielding."]
    },
    {
        "tag": "tennis",
        "patterns": ["What is tennis?", "Tell me about tennis", "How is tennis played?"],
        "responses": ["Tennis is a racket sport that can be played individually against a single opponent or between two teams of two players each. Each player uses a tennis racket to strike a hollow rubber ball covered with felt over or around a net and into the opponent's court."]
    },{
        "tag": "tennis_rules",
        "patterns": ["What are the rules of tennis?", "Explain the rules of tennis", "How do you play tennis?"],
        "responses": ["Tennis is played between two players (singles) or two teams of two players each (doubles). The objective is to hit the ball over the net and into the opponent's court in such a way that the opponent cannot return it. A match is played in sets, and the player/team that wins the majority of sets wins the match."]
    },
    {
        "tag": "athletics",
        "patterns": ["What is athletics?", "Tell me about athletics", "What sports are included in athletics?"],
        "responses": ["Athletics is a collection of sporting events that involve competitive running, jumping, throwing, and walking. The most common types of athletics competitions are track and field, road running, cross country running, and race walking."]
    },
    {
        "tag": "baseball",
        "patterns": ["What is baseball?", "Tell me about baseball", "How is baseball played?"],
        "responses": ["Baseball is a bat-and-ball game played between two opposing teams who take turns batting and fielding. The game proceeds when a player on the fielding team, called the pitcher, throws a ball that a player on the batting team tries to hit with a bat."]
    },{
        "tag": "baseball_rules",
        "patterns": ["What are the rules of baseball?", "Explain the rules of baseball", "How do you play baseball?"],
        "responses": ["Baseball is played between two teams of nine players each. The objective is to score runs by hitting the ball and running around four bases arranged in a diamond. The fielding team tries to get the batting team out through various plays such as catching the ball or tagging the runner."]
    },
    {
        "tag": "golf",
        "patterns": ["What is golf?", "Tell me about golf", "How is golf played?"],
        "responses": ["Golf is a club-and-ball sport in which players use various clubs to hit balls into a series of holes on a course in as few strokes as possible."]
    },{
        "tag": "golf_rules",
        "patterns": ["What are the rules of golf?", "Explain the rules of golf", "How do you play golf?"],
        "responses": ["Golf is played on a course with 9 or 18 holes. Players use different clubs to hit a ball from the tee into the hole in as few strokes as possible. The player with the lowest total strokes at the end of the round wins. Each hole has a par, which is the expected number of strokes to complete the hole."]
    },
    {
        "tag": "swimming",
        "patterns": ["What is swimming?", "Tell me about swimming", "How is swimming done?"],
        "responses": ["Swimming is an individual or team racing sport that requires the use of one's entire body to move through water. The sport takes place in pools or open water."]
    },{
        "tag": "swimming_rules",
        "patterns": ["What are the rules of swimming?", "Explain the rules of swimming", "How do you compete in swimming?"],
        "responses": ["Swimming competitions are categorized into different strokes: freestyle, backstroke, breaststroke, and butterfly. Races are held over various distances, and swimmers must follow specific techniques for each stroke. The swimmer who touches the wall first at the end of the distance wins the race."]
    },
    {
        "tag": "rugby",
        "patterns": ["What is rugby?", "Tell me about rugby", "How is rugby played?"],
        "responses": ["Rugby is a sport that consists of two teams with 15 players each (in rugby union) or 13 players each (in rugby league), who compete to carry or kick a ball over the opponent's goal line to score points."]
    },{
        "tag": "rugby_rules",
        "patterns": ["What are the rules of rugby?", "Explain the rules of rugby", "How do you play rugby?"],
        "responses": ["Rugby is played between two teams of 15 (rugby union) or 13 (rugby league) players. The objective is to score points by carrying, passing, or kicking the ball over the opponent's goal line. Points can be scored through tries, conversions, penalty kicks, and drop goals."]
    },
    {
        "tag": "boxing",
        "patterns": ["What is boxing?", "Tell me about boxing", "How is boxing done?"],
        "responses": ["Boxing is a combat sport in which two people, usually wearing protective gloves and other protective equipment such as hand wraps and mouthguards, throw punches at each other for a predetermined amount of time in a boxing ring."]
    }, {
        "tag": "boxing_rules",
        "patterns": ["What are the rules of boxing?", "Explain the rules of boxing", "How do you compete in boxing?"],
        "responses": ["Boxing matches consist of a specified number of three-minute rounds, with one-minute intervals between them. Boxers score points for every punch they land on their opponent's torso or head. The match can be won by knockout, technical knockout, or by judges' decision based on points."]
    },
    {
        "tag": "soccer",
        "patterns": ["What is soccer?", "Tell me about soccer", "How is soccer played?"],
        "responses": ["Soccer, also known as football outside the US and Canada, is a team sport played with a spherical ball between two teams of 11 players. The game is played on a rectangular field with a goal at each end."]
    }, {
        "tag": "soccer_rules",
        "patterns": ["What are the rules of soccer?", "Explain the rules of soccer", "How do you play soccer?"],
        "responses": ["Soccer is played between two teams of 11 players on a rectangular field. The objective is to score by getting the ball into the opposing goal. The game is played in two halves of 45 minutes each. Players can use any part of their body except their hands and arms to move the ball, while the goalkeepers are allowed to use their hands within the penalty area."]
    },
    {
        "tag": "cycling",
        "patterns": ["What is cycling?", "Tell me about cycling", "How is cycling done?"],
        "responses": ["Cycling, also called biking or bicycling, is the use of bicycles for transport, recreation, exercise or sport. People engaged in cycling are referred to as 'cyclists', 'bikers', or less commonly, as 'bicyclists'."]
    }, {
        "tag": "cycling_rules",
        "patterns": ["What are the rules of cycling?", "Explain the rules of cycling", "How do you compete in cycling?"],
        "responses": ["Cycling competitions include road races, time trials, and track cycling. Cyclists must follow the specific rules of the race, such as staying on the designated route, completing the set distance, and avoiding unsportsmanlike conduct. The winner is the first to cross the finish line or the fastest in a time trial."]
    },
    {
        "tag": "volleyball",
        "patterns": ["What is volleyball?", "Tell me about volleyball", "How is volleyball played?"],
        "responses": ["Volleyball is a team sport in which two teams of six players are separated by a net. Each team tries to score points by grounding a ball on the other team's court under organized rules."]
    }, {
        "tag": "volleyball_rules",
        "patterns": ["What are the rules of volleyball?", "Explain the rules of volleyball", "How do you play volleyball?"],
        "responses": ["Volleyball is played with two teams of six players each. The objective is to send the ball over the net and ground it on the opponent's court. Each team is allowed three touches to return the ball. A match is played in sets, and the team that wins the majority of sets wins the match."]
    },
    {
        "tag": "skiing",
        "patterns": ["What is skiing?", "Tell me about skiing", "How is skiing done?"],
        "responses": ["Skiing is a means of transport using skis to glide on snow. Variations of purpose include basic transport, a recreational activity, or a competitive winter sport."]
    },{
        "tag": "skiing_rules",
        "patterns": ["What are the rules of skiing?", "Explain the rules of skiing", "How do you compete in skiing?"],
        "responses": ["Skiing competitions include alpine skiing, cross-country skiing, and freestyle skiing. Each type has specific rules for the course, technique, and timing. In alpine skiing, competitors race down a set course and the fastest time wins. Cross-country skiing involves long-distance races with various techniques."]
    },
    {
        "tag": "gymnastics",
        "patterns": ["What is gymnastics?", "Tell me about gymnastics", "What does gymnastics involve?"],
        "responses": ["Gymnastics is a sport that includes exercises requiring balance, strength, flexibility, agility, coordination, and endurance. The movements involved in gymnastics contribute to the development of the arms, legs, shoulders, back, chest, and abdominal muscle groups."]
    },{
        "tag": "gymnastics_rules",
        "patterns": ["What are the rules of gymnastics?", "Explain the rules of gymnastics", "How do you compete in gymnastics?"],
        "responses": ["Gymnastics competitions are divided into events like floor exercise, pommel horse, rings, vault, parallel bars, and horizontal bar for men, and vault, uneven bars, balance beam, and floor exercise for women. Athletes perform routines judged on difficulty, execution, and artistic impression."]
    },
    {
        "tag": "badminton",
        "patterns": ["What is badminton?", "Tell me about badminton", "How is badminton played?"],
        "responses": ["Badminton is a racquet sport played using racquets to hit a shuttlecock across a net. It can be played as singles (one player per side) or doubles (two players per side)."]
    }, {
        "tag": "badminton_rules",
        "patterns": ["What are the rules of badminton?", "Explain the rules of badminton", "How do you play badminton?"],
        "responses": ["Badminton is played on a rectangular court divided by a net. The objective is to score points by hitting the shuttlecock over the net and into the opponent's court. Each side can only hit the shuttlecock once before it passes over the net. A match is the best of three games to 21 points."]
    },
    {
        "tag": "fencing",
        "patterns": ["What is fencing?", "Tell me about fencing", "How is fencing done?"],
        "responses": ["Fencing is a group of three related combat sports. The three disciplines in modern fencing are the foil, the épée, and the sabre; winning points are made through the weapon's contact with an opponent."]
    },{
        "tag": "fencing_rules",
        "patterns": ["What are the rules of fencing?", "Explain the rules of fencing", "How do you compete in fencing?"],
        "responses": ["Fencing matches are contested on a strip or 'piste', with points awarded for making contact with the opponent using the weapon. Each discipline has its own rules regarding the target area and the method of scoring. Bouts are won by reaching a set number of points or having the highest score within the time limit."]
    },
    {
        "tag": "table_tennis",
        "patterns": ["What is table tennis?", "Tell me about table tennis", "How is table tennis played?"],
        "responses": ["Table tennis, also known as ping-pong, is a sport in which two or four players hit a lightweight ball, also known as the ping-pong ball, back and forth across a table using small bats."]
    }, {
        "tag": "table_tennis_rules",
        "patterns": ["What are the rules of table tennis?", "Explain the rules of table tennis", "How do you play table tennis?"],
        "responses": ["Table tennis is played on a hard table divided by a net. The objective is to score points by hitting the ball over the net and onto the opponent's side of the table in such a way that the opponent cannot return it. Matches are typically best of five or seven games, with each game played to 11 points."]
    },
    {
        "tag": "surfing",
        "patterns": ["What is surfing?", "Tell me about surfing", "How is surfing done?"],
        "responses": ["Surfing is a surface water sport in which the wave rider, referred to as a surfer, rides on the forward or deep face of a moving wave, which usually carries the surfer towards the shore."]
    }, {
        "tag": "surfing_rules",
        "patterns": ["What are the rules of surfing?", "Explain the rules of surfing", "How do you compete in surfing?"],
        "responses": ["Surfing competitions are judged based on the difficulty and execution of maneuvers performed on the wave. Surfers are scored on a scale from 1 to 10, with the top two waves counting towards their total score. The surfer with the highest total score at the end of the heat wins."]
    },
    {
        "tag": "hockey",
        "patterns": ["What is hockey?", "Tell me about hockey", "How is hockey played?"],
        "responses": ["Hockey is a sport in which two teams play against each other by trying to maneuver a ball or a puck into the opponent's goal using a hockey stick. There are many types of hockey such as field hockey, ice hockey, and roller hockey."]
    },{
        "tag": "hockey_rules",
        "patterns": ["What are the rules of hockey?", "Explain the rules of hockey", "How do you play hockey?"],
        "responses": ["Hockey is played between two teams of six players each on an ice rink for ice hockey or a field for field hockey. The objective is to score by getting the puck or ball into the opposing goal. The game consists of three periods, and the team with the most goals at the end wins."]
    }
]

# Preprocess the data: lemmatize and remove stop words
def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(words)

patterns = [preprocess_text(pattern) for intent in intents for pattern in intent['patterns']]
tags = [intent['tag'] for intent in intents for pattern in intent['patterns']]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = preprocess_text(input_text)
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Tkinter GUI
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sport Chatbot")
        
        # Chat area
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=60, height=20, padx=10, pady=10, bg='light grey', fg='blue')
        self.chat_area.grid(row=0, column=0, columnspan=2)
        
        # User input field
        self.user_input = tk.Entry(root, width=40, bg='light grey', fg='blue')
        self.user_input.grid(row=1, column=0, padx=10, pady=10)
        self.user_input.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.send_message, bg='blue', fg="#fff")
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
       

    def send_message(self, event=None):
        user_message = self.user_input.get()
        if user_message.strip():
            self.display_message(f'\nYou: {user_message}')
            response = chatbot(user_message)
            self.display_message(f'Chatbot: {response}')
            self.user_input.delete(0, tk.END)
            if user_message.lower() in ['bye', 'goodbye', 'exit', 'quit']:
               self.display_message("Chatbot: Goodbye! Have a great day!")
               self.root.quit()

    def display_message(self, message):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, message + '\n')
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

def main():
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()





