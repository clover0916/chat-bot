import discord
import MeCab
import json

client = discord.Client()
json_open = open('./config.json', 'r')
config = json.load(json_open)

async def mecab(message, args):
  data = ' '.join(args[1:])
  mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd').parse(data)
  await message.channel.send('```' + mecab + '```')

@client.event
async def on_ready():
  print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
  if not message.content.startswith(config['prefix']) and message.author == client.user: 
    return

  args = message.content[len(config['prefix']):].strip().split(" ")

  if message.content.startswith('cm!mecab'):
    await mecab(message, args)

client.run(config['token'])