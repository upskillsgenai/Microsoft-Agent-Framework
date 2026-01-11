import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv # add this in original file

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field
from typing import Annotated

# Load environment variables
load_dotenv() # add this in original file

async def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    script_dir = Path(__file__).parent
    file_path = script_dir / 'data.txt'
    with file_path.open('r') as file:
        data = file.read() + "\n"

    user_prompt = input(
        f"Here is the expenses data in your file:\n\n{data}\n\nWhat would you like me to do with it?\n\n"
    )

    await process_expenses_data(user_prompt, data)


async def process_expenses_data(prompt, expenses_data):
    async with (
        AzureCliCredential() as credential,
        ChatAgent(
            chat_client=AzureAIAgentClient(
                credential=credential,   # add this in original file
                project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"), # add this in original file
                model_deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME"), # add this in original file
            ),
            name="expenses_agent",
            instructions="""You are an AI assistant for expense claim submission.
                            When a user submits expenses data and requests an expense claim, 
                            use the plug-in function to send an email to expenses@contoso.com 
                            with the subject 'Expense Claim' and a body that contains itemized 
                            expenses with a total. Then confirm to the user that you've done so.""",
            tools=send_email,
        ) as agent,
    ):
        try:
            prompt_messages = [f"{prompt}: {expenses_data}"]
            response = await agent.run(prompt_messages)
            print(f"\n# Agent:\n{response}")
        except Exception as e:
            print("Error while running agent:", e)


def send_email(
    to: Annotated[str, Field(description="Who to send the email to")],
    subject: Annotated[str, Field(description="The subject of the email.")],
    body: Annotated[str, Field(description="The text body of the email.")],
):
    print("\nTo:", to)
    print("Subject:", subject)
    print(body, "\n")


if __name__ == "__main__":
    asyncio.run(main())

