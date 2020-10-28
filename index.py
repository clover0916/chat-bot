import discord
import MeCab
import json

client = discord.Client()
json_open = open('./config.json', 'r')
config = json.load(json_open)

async def mecab(message, args):
  mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd').parse(args)
  lines = mecab.split('\n')
  items = (re.split('[\t]',line) for line in lines)
  for item in items:
    await message.channel.send(item)

@client.event
async def on_ready():
  print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
  if not message.content.startsWith(config.prefix) and message.author == client.user: 
    return

  args = message.content[len(config.prefix):].strip().split(" ")

  if message.content.startswith('cm!mecab'):
    await mecab(message, args)

client.run(config.token)