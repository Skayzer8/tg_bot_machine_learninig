
import json
import random
import re

import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from telegram import Update
from telegram.ext import Filters, MessageHandler, Updater


DISTANSE_LVL = 0.2
config_file = open('big_bot_config.json', 'r')

BOT_CONFIG = json.load(config_file)
# BOT_KEY = BOT API KEY


X = []
y = []

for name, data in BOT_CONFIG['intents'].items():
    for example in data['examples']:
        X.append(example)
        y.append(name)

vectorizer = CountVectorizer()
vectorizer.fit(X)
vecX = vectorizer.transform(X)

model = RandomForestClassifier(n_estimators=500, min_samples_split=3)
model.fit(vecX, y)
y_pred = model.predict(vecX)


def filter(client_msg):
    """Очищение сообщения пользователя от лишних символов."""
    ALPHABET = re.sub(r'[^ a-zA-Zа-яА-ЯёЁ]', '', client_msg)
    result = [simb for simb in client_msg if simb in ALPHABET]
    return ''.join(result)


def match(client_msg, example):
    """Сравнение текстовой близости сообщения
    пользователя с ключами из словаря."""
    client_msg = filter(client_msg.lower())
    example = example.lower()
    distance = nltk.edit_distance(client_msg, example) / len(example)
    return distance < DISTANSE_LVL


def get_intent(client_msg):
    """Поиск совпадений в базе."""
    for intent in BOT_CONFIG['intents']:
        for example in BOT_CONFIG['intents'][intent]['examples']:
            if match(client_msg, example):
                return intent


def bot(client_msg):
    intent = get_intent(client_msg)

    if not intent:
        transformed_text = vectorizer.transform([client_msg])
        intent = model.predict(transformed_text)[0]

    if intent:
        return random.choice(BOT_CONFIG['intents'][intent]['responses'])

    return random.choice(BOT_CONFIG['intents'][intent]['responses'])


def botStartMessaging(update: Update, context):
    """Функция отвечающая за общение бота с пользователем."""
    msg = update.message.text
    reply = bot(msg)
    update.message.reply_text(reply)


Bot_upd = Updater(BOT_KEY)
handler = MessageHandler(Filters.text, botStartMessaging)
Bot_upd.dispatcher.add_handler(handler)
Bot_upd.start_polling()
Bot_upd.idle()
