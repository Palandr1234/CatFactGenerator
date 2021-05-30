import telebot
import argparse
import pickle
from model import Model
from preprocess_data import Dataset, normalizeString
import random
import torch
from telebot import types
from emoji import emojize
import numpy as np
import re
from test import predict


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('token', type=str)
parser.add_argument('--dataset', type=str, default="output/vocabulary.pkl")
parser.add_argument('--model', type=str, default="state_dict_model.pt")
args = parser.parse_args()

bot = telebot.TeleBot(args.token)


# generate the beginning of the text
# dataset - dataset to be used
def choose_beginning(dataset):
    # find all possible beginnigs for the text
    indices = [(i + 1) % len(dataset.words) for i, x in enumerate(dataset.words) if x == "EOS"]
    # generate the length of the beginning
    length = random.randint(1, 5)
    # choose the beginning
    idx = random.choice(indices)
    return ' '.join(dataset.words[idx:idx+length])


# function when the user starts  using the bot
@bot.message_handler(commands=['start'])
def hello(message):
    # generate the keyboard
    command_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    fact_button = types.KeyboardButton('fact')
    help_button = types.KeyboardButton('help')
    command_keyboard.add(fact_button, help_button)
    # send the hello message
    bot.send_message(message.from_user.id, text="Hello! I am the bot for generating facts about cats",
                     reply_markup=command_keyboard)


# generate the fact if the user types the command /fact
@bot.message_handler(content_types=['text'], commands=['fact'])
def generate_fact(message):
    # generate the beginning of the text
    initial_text = choose_beginning(dataset)
    # generate the text
    text = ' '.join(predict(dataset, model, initial_text))
    # normalize text
    text = re.sub(r'\s([:?.!,;)](?:\s|$))', r'\1', text)
    text = re.sub(r'((?:\s|$)[(])\s', r'\1', text)
    # add the keyboard
    like_keyboard = types.InlineKeyboardMarkup()
    like_button = types.InlineKeyboardButton(text=emojize(':thumbs_up:'), callback_data='like')
    dislike_button = types.InlineKeyboardButton(text=emojize(':thumbs_down:'), callback_data="dislike")
    like_keyboard.add(like_button, dislike_button)
    # send the message with the fact
    bot.send_message(message.from_user.id, text=text, reply_markup=like_keyboard)


# check whether user put the like or dislike
@bot.callback_query_handler(func=lambda call: True)
def is_like(call):
    # if it is like then send "thanks for like" and save the fact that was liked into the dataset
    if call.data == 'like':
        bot.send_message(call.message.chat.id, "Thanks for like")
        with open('output/new.txt', 'a') as f:
            f.write(call.message.text)
            f.write('\n')
    # if it is dislike then send "ok"
    elif call.data == 'dislike':
        bot.send_message(call.message.chat.id, "Ok")


# process messages other than /start and /fact command
@bot.message_handler(content_types=['text'])
def process_text(message):
    if message.text == 'fact':
        generate_fact(message)
    # if the message is help then send the help message
    elif message.text == 'help':
        bot.send_message(message.from_user.id, text='Hello! I am the bot for generating facts about cats. Send the'
                                                    ' command /fact for generating the fact or press the button "Fact"')
    # otherwise send that we do not understand the user
    else:
        bot.send_message(message.from_user.id, text="I don't understand you")


if __name__ == '__main__':
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    PATH = args.model
    # load the model
    model = Model(dataset)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    bot.polling(none_stop=True, interval=0)
