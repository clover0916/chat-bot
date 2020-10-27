import discord

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('cm!hello'):
        await message.channel.send('Hello!')

client.run('NzcwNjMxNjEwOTQ5MTczMjg4.X5gYnQ.EnW08GySiuOlYUyHSJeWguRvCBE')