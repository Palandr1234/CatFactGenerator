import telebot
import argparse
import pickle
from test import predict
from model import Model
from preprocess_data import Dataset
import random
import torch
from telebot import types
from emoji import emojize
import re


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('token', type=str)
parser.add_argument('--dataset', type=str, default="output/vocabulary.pkl")
parser.add_argument('--model', type=str, default="state_dict_model.pt")
args = parser.parse_args()

bot = telebot.TeleBot(args.token)


def choose_beginning(dataset):
    indices = [(i + 1) % len(dataset.words) for i, x in enumerate(dataset.words) if x == "EOS"]
    length = random.randint(1, 5)
    idx = random.choice(indices)
    return ' '.join(dataset.words[idx:idx+length])


@bot.message_handler(commands=['start'])
def hello(message):
    command_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    fact_button = types.KeyboardButton('fact')
    help_button = types.KeyboardButton('help')
    command_keyboard.add(fact_button, help_button)
    bot.send_message(message.from_user.id, text="Hello! I am the bot for generating facts about cats",
                     reply_markup=command_keyboard)


@bot.message_handler(content_types=['text'], commands=['fact'])
def generate_fact(message):
    initial_text = choose_beginning(dataset)
    text = ' '.join(predict(dataset, model, initial_text))
    text = re.sub(r'\s([:?.!,;)](?:\s|$))', r'\1', text)
    text = re.sub(r'((?:\s|$)[(])\s', r'\1', text)
    like_keyboard = types.InlineKeyboardMarkup()
    like_button = types.InlineKeyboardButton(text=emojize(':thumbs_up:'), callback_data='like')
    dislike_button = types.InlineKeyboardButton(text=emojize(':thumbs_down:'), callback_data="dislike")
    like_keyboard.add(like_button, dislike_button)
    bot.send_message(message.from_user.id, text=text, reply_markup=like_keyboard)


@bot.callback_query_handler(func=lambda call: True)
def is_like(call):
    if call.data == 'like':
        bot.send_message(call.message.chat.id, "Thanks for like")
        with open('output/new.txt', 'a') as f:
            f.write(call.message.text)
            f.write('\n')
    elif call.data == 'dislike':
        bot.send_message(call.message.chat.id, "Ok")


@bot.message_handler(content_types=['text'])
def process_text(message):
    if message.text == 'fact':
        generate_fact(message)
    elif message.text == 'help':
        bot.send_message(message.from_user.id, text='Hello! I am the bot for generating facts about cats. Send the'
                                                    ' command /fact for generating the fact or press the button "Fact"')
    else:
        bot.send_message(message.from_user.id, text="I don't understand you")


if __name__ == '__main__':
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    PATH = args.model
    model = Model(dataset)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    bot.polling(none_stop=True, interval=0)
