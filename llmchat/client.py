import asyncio
import discord
import requests
from typing import Union, Any
from discord import app_commands
from discord.interactions import Interaction
import openai
from aiohttp import ClientSession
import llmchat.ui_extensions

from llmchat.config import Config
from llmchat.logger import logger, console_handler, color_formatter
from llmchat.persistence import PersistentData

from llmchat.llm_sources import LLMSource

class DiscordClient(discord.Client):
    config: Config
    llm: LLMSource = None
    db: PersistentData

    def __init__(self, config: Config, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        if not self.config.can_interact_with_channel_id(-1) and not self.config.discord_active_channels:
            raise Exception(
                "There aren't any active channels specified in your config.ini! Please add text/voice channel ids to Discord.active_channels (seperated by commas),"
                " or set Discord.active_channels to \"all\" and the bot will interact with all channels.")

        if self.config.can_interact_with_channel_id(-1):
            logger.warn("Discord.active_channels = \"all\", bot will interact with every channel!")

        intents = discord.Intents.default()
        intents.message_content = True
        super(DiscordClient, self).__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        self.tree.add_command(
            app_commands.Command(
                name="info",
                description="Prints some information about the bot.",
                callback=self.print_info,
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="configure",
                description="Configure the chatbot.",
                callback=self.show_configure,
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="your_identity",
                description="Sets your identity.",
                callback=self.set_your_identity,
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="avatar",
                description="Sets the chatbot's avatar.",
                callback=self.set_avatar,
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="purge",
                description="Deletes all messages in channel. [DANGEROUS!]",
                callback=self.purge_channel,
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="model", description="Allows you to change the LLM model.", callback=self.set_model
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="retry",
                description="Retries the last message.",
                callback=self.retry_last_message,
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="system",
                description="Sends a message as the system role. Only avaliable for >= GPT3.5.",
                callback=self.send_system,
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="message_context_count",
                description="Sets the message context count. Default is 10.",
                callback=self.set_message_context_count,
            )
        )
        self.tree.add_command(
            app_commands.Command(
                name="reload_config",
                description="Reloads the settings from config.ini.",
                callback=self.reload_config,
            )
        )

        self.run(
            self.config.discord_bot_api_key,
            log_handler=console_handler,
            log_formatter=color_formatter,
        )

    async def setup_llm(self):
        logger.info(f"LLM: {self.config.bot_llm}")
        params = [self, self.config, self.db]
        if self.config.bot_llm == "openai":
            from llmchat.llm_sources.oai import OpenAI
            self.llm = OpenAI(*params)
        elif self.config.bot_llm == "llama":
            from llmchat.llm_sources.llama import LLaMA
            self.llm = LLaMA(*params)
        else:
            logger.critical(f"Unknown LLM: {self.config.bot_llm}")

        await self.change_presence(activity=discord.Game(name=self.llm.current_model_name))
        logger.info(f"Current model: {self.llm.current_model_name}")

    async def reload_config(self, ctx: Interaction):
        await ctx.response.defer()

        try:
            prev_llm, = self.config.bot_llm
            self.config.load()
            # manually load new settings if necessary
            if prev_llm != self.config.bot_llm:
                await self.setup_llm()

            self.llm.on_config_reloaded()

            logger.info("Config reloaded.")
            followup: discord.WebhookMessage = await ctx.followup.send(content="Config reloaded.")
            await followup.delete(delay=3)
        except BaseException as e:
            logger.error(f"Error reloading config: {str(e)}")
            followup: discord.WebhookMessage = await ctx.followup.send(content=f"Error reloading config: ```{str(e)}```")
            await followup.delete(delay=5)

    async def retry_last_message(self, ctx: Interaction):
        history_item = self.db.last

        await ctx.response.defer()

        if not history_item:
            response = await self.llm.generate_response(ctx.user)
            sent_message = await self.send_message(response, ctx.followup)
            await self.store_embedding((ctx.user.id, response, sent_message[0].id))
            self.db.append(sent_message[0], override_content=response)
            return

        author_id, content, message_id = history_item
        last_message: discord.Message = await ctx.channel.fetch_message(message_id)

        if author_id != self.user.id:
            # not from me
            response = await self.llm.generate_response(ctx.user)
            sent_message = await self.send_message(response, ctx.followup)
            await self.store_embedding((ctx.user.id, response, sent_message[0].id))
            self.db.append(sent_message[0], override_content=response)
        else:
            delete_me = await ctx.followup.send(content="Retrying...", silent=True)
            await delete_me.delete()
            await last_message.edit(content="*Retrying...*")
            self.db.remove(last_message.id)
            response = await self.llm.generate_response(ctx.user)

            if len(response) < 2000:
                await last_message.edit(content=response)
            else:
                await last_message.delete()
                last_message = await self.send_message(response, ctx.channel)
                last_message = last_message[0]

            await self.store_embedding((ctx.user.id, response, last_message.id))
            self.db.append(last_message, override_content=response)

    async def set_message_context_count(self, ctx: Interaction, count: int):
        self.config.llm_context_messages_count = count
        await ctx.response.send_message(
            f"Set message context count to {count}", delete_after=3
        )

    async def set_avatar(self, ctx: Interaction, url: str):
        r = requests.get(url, stream=True)
        await self.user.edit(avatar=r.content)
        await ctx.response.send_message(f"Avatar set!", delete_after=3)

    async def send_message(self, text: str, channel: Union[discord.TextChannel, discord.Webhook]) -> list[discord.Message]:
        # message splitting
        all_messages = []
        initial_message: Union[discord.Message, None] = None
        char_limit = 2000
        if len(text) < char_limit:
            initial_message = await channel.send(content=text)
            all_messages.append(initial_message)
        else:
            split_msg = None
            chunks = [text[i:i + char_limit] for i in range(0, len(text), char_limit)]
            for c in chunks:
                split_msg = await channel.send(content=c, reference=split_msg)
                all_messages.append(split_msg)
                if not initial_message:
                    initial_message = split_msg
                await asyncio.sleep(0.5)
        return all_messages

    async def print_info(self, ctx: Interaction):
        await ctx.response.defer()

        name, identity = (None, None)
        _identity = self.db.get_identity(ctx.user.id)
        if _identity:
            name, identity = _identity

        embed = discord.Embed(title="Chatbot info")
        llm_str = f"**{self.config.bot_llm}**: {self.llm.current_model_name}\n"
        llm_str += f"⚙️ Temperature: {self.config.llm_temperature}\n"
        llm_str += f"⚙️ Presence penalty: {self.config.llm_presence_penalty}\n"
        llm_str += f"⚙️ Frequency penalty: {self.config.llm_frequency_penalty}\n"
        llm_str += f"⚙️ Context history count: {self.config.llm_context_messages_count}\n"
        llm_str += f"⚙️ Max tokens: {'Unlimited' if self.config.llm_max_tokens == 0 else self.config.llm_max_tokens}\n"

        embed.add_field(name="📝 LLM", value=llm_str, inline=False)
        embed.add_field(name="\u200B", value="", inline=False)  # seperator

        embed.add_field(name="📛 Name", value=self.config.bot_name, inline=False)
        embed.add_field(name="✒️ Description", value=self.config.bot_identity, inline=False)
        embed.add_field(name="🎗️ Reminder",
                        value=self.llm._insert_wildcards(self.config.bot_reminder, (name, identity)) if self.config.bot_reminder else "*Not set!*",
                        inline=False)
        embed.add_field(name="✍️ Initial prompt", value=self.llm._insert_wildcards(self.config.bot_initial_prompt, (name, identity)) if self.config.bot_initial_prompt else "*Not set!*", inline=False)
        embed.add_field(name="\u200B", value="", inline=False)  # seperator

        embed.add_field(
            name="🫵 Your info (set this information with /your_identity)",
            value="",
            inline=False,
        )
        embed.add_field(
            name="📛 Name", value=name if name is not None else "*Not set!*")
        embed.add_field(
            name="✒️ Description", value=identity if identity is not None else "*Not set!*"
        )

        await ctx.followup.send(embed=embed)

    async def purge_channel(self, ctx: Interaction):
        await ctx.response.send_message(f"Channel purged!", delete_after=3)
        await ctx.channel.purge()
        self.db.clear()

    async def set_model(self, ctx: Interaction):

        await ctx.response.defer()
        async def llm_callback(ctx: Interaction):
            try:
                model = ctx.data["values"][0]
                self.llm.set_model(model)
                await self.change_presence(activity=discord.Game(name=model))
                await ctx.response.edit_message(content=f"Model changed to *{self.llm.current_model_name}*", embed=None, view=None, delete_after=3)
            except Exception as e:
                logger.error(f"Exception thrown while LLM model: {str(e)}")
                await ctx.response.edit_message(content=f"Exception thrown while setting LLM model:\n```{str(e)}```", embed=None, view=None, delete_after=5)

        try:
            def on_exception(e):
                raise e

            view = discord.ui.View()
            view.add_item(llmchat.ui_extensions.PaginationDropdown(options=await self.llm.list_models(), callback=llm_callback, on_exception=on_exception))
            await ctx.followup.send(content="Select an LLM model:", view=view)
        except Exception as e:
            logger.error(f"Exception thrown while constructing model pickers: {str(e)}")
            exc_message = await ctx.followup.send(content=f"Exception thrown while constructing model pickers:\n```{str(e)}```")
            await exc_message.delete(delay=5)


    async def send_system(self, ctx: Interaction, message: str):
        if self.config.bot_llm == "openai" and self.llm.use_chat_completion:
            await ctx.response.send_message(f"**System**: {message}")
            self.db.system(message, ctx.id)
        else:
            await ctx.response.send_message(
                "Error: System messages are only supported in OpenAI models, gpt-3.5-turbo and newer.",
                delete_after=5,
            )

    async def set_your_identity(self, ctx: Interaction):
        this = self

        name, desc = self.db.get_identity(ctx.user.id) or (None, None)

        class IdentityModal(discord.ui.Modal):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

                self.add_item(
                    discord.ui.TextInput(
                        label="Name",
                        custom_id="name",
                        placeholder=ctx.user.display_name,
                        default=name,
                    )
                )
                self.add_item(
                    discord.ui.TextInput(
                        label="Description",
                        placeholder=f"{ctx.user.display_name} is a discord user who doesn't know what to put for their description.",
                        custom_id="description",
                        style=discord.TextStyle.paragraph,
                        default=desc,
                    )
                )

            async def on_submit(self, interaction: Interaction):
                this.db.set_identity(
                    ctx.user.id, self.children[0].value, self.children[1].value
                )
                await interaction.response.send_message("Changes committed.", delete_after=3)

        modal = IdentityModal(title=f"Edit {ctx.user.display_name}'s identity")

        await ctx.response.send_modal(modal)

    async def show_configure(self, ctx: Interaction):
        this = self

        class ConfigureModal(discord.ui.Modal):
            def __init__(self, page=1, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

                self.add_item(
                    discord.ui.TextInput(
                        label="Name",
                        custom_id="name",
                        placeholder="Jackson",
                        default=this.config.bot_name,
                    )
                )
                self.add_item(
                    discord.ui.TextInput(
                        label="Description",
                        custom_id="description",
                        placeholder="Jackson is a cool bot. He types in all lowercase because he's edgy.",
                        style=discord.TextStyle.paragraph,
                        default=this.config.bot_identity,
                    )
                )
                self.add_item(
                    discord.ui.TextInput(
                        label="Reminder",
                        custom_id="reminder",
                        placeholder="A short context clue to remind the bot to stay in character.",
                        style=discord.TextStyle.paragraph,
                        default=this.config.bot_reminder,
                        required=False,
                    )
                )
                self.add_item(
                    discord.ui.TextInput(
                        label="Initial prompt",
                        custom_id="initial",
                        placeholder="The bot's initial prompt. (Example: You are a helpful assistant named {bot_name}.)",
                        style=discord.TextStyle.paragraph,
                        default=this.config.bot_initial_prompt,
                        required=False,
                    )
                )

            async def on_submit(self, interaction: Interaction):
                this.config.bot_name = self.children[0].value
                await interaction.guild.me.edit(nick=this.config.bot_name)
                this.config.bot_identity = self.children[1].value
                this.config.bot_reminder = self.children[2].value
                this.config.bot_initial_prompt = self.children[3].value

                await interaction.response.send_message("Changes committed.", delete_after=3)

        modal = ConfigureModal(title="Configure chatbot")

        await ctx.response.send_modal(modal)

    async def on_ready(self):
        logger.info(f"Logged in as {self.user}")
        await self.wait_until_ready()

        await self.change_presence(activity=discord.Game(name="Loading..."))

        self.db: PersistentData = PersistentData(self)
        await self.setup_llm()

        await self.tree.sync()
        logger.info("Initialization complete.")

    async def store_embedding(self, message: tuple[int, str, int]):
        author_id, content, message_id = message
        if self.config.openai_use_embeddings and self.llm.is_openai:
            async with ClientSession() as s:
                openai.aiosession.set(s)
                embedding = await openai.Embedding.acreate(api_base=self.config.openai_reverse_proxy_url, input=content, model="text-embedding-ada-002")
                self.db.add_embedding(message, embedding['data'][0]['embedding'])
                logger.debug("Added embedding for message " + str(message_id))

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):
        self.db.remove(payload.message_id)

        message = payload.cached_message
        if not message:
            logger.warn(
                "The bot was unable to look for any messages that may have been split from this one! Make sure to delete the parent message to remove it from the history!")
            return

        channel = self.get_channel(payload.channel_id)
        while message.reference:
            reference = await channel.fetch_message(message.reference.message_id)
            self.db.remove(reference.id)
            await reference.delete()
            logger.debug(f"Deleted reference message: {reference.id}")
            message = reference

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent):
        self.db.edit(payload.message_id, payload.data["content"])

        if payload.cached_message:
            self.db.remove_embedding(payload.cached_message.id)  # remove existing
            await self.store_embedding((payload.cached_message.author.id, payload.data["content"], payload.cached_message.id))  # regenerate

    async def on_message(self, message: discord.Message):
        if message.author.id == self.user.id \
                or not self.config.can_interact_with_channel_id(message.channel.id) \
                or not message.content:
            # from me or not allowed in channel
            return

        self.db.append(message)
        await self.store_embedding((message.author.id, message.content, message.id))

        async with message.channel.typing():
            try:
                response = await self.llm.generate_response(invoker=message.author)
            except Exception as e:
                view = discord.ui.View()
                retry_btn = discord.ui.Button(label="Retry")

                async def _retry(interaction: Interaction):
                    await interaction.message.delete(delay=2)
                    await self.on_message(message)

                retry_btn.callback = _retry
                view.add_item(retry_btn)

                await message.channel.send(f"Exception thrown while trying to generate message:\n```{str(e)}```",
                                           view=view)

                # since it failed remove the message
                self.db.remove(message.id)
                raise e

        logger.debug(f"Response: {response}")

        sent_message = await self.send_message(response, message.channel)
        sent_message = sent_message[0]

        assert sent_message
        self.db.append(sent_message, override_content=response)

        await self.store_embedding((self.user.id, response, sent_message.id))
