from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, load_prompt
from langchain.schema import BaseOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain_core.prompts.few_shot import FewShotPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import pandas as pd
import PRIVATE


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=PRIVATE.key
)


df = pd.read_csv('/content/res_info.csv')
restaurant = df[['MCT_NM', 'name']]

results = []

for i, MCT_NM, name in restaurant.itertuples(index=True):
    question = f"'{MCT_NM}'과 '{name}'이 같은 업체일까?"

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "너는 음식점이 같은 음식점인지 확인하고 True 혹은 False로만 답변을 하는 봇이야."),
            ("human", "True 혹은 False인지 말을 해. 다른 내용은 절대 말하지 마.\n" + question),
        ]
    )

    chain = final_prompt | llm
    answer = chain.invoke({"question": question})
    final_answer = answer.content

    results.append(final_answer)

    print(f"{i} |\t{final_answer}\t | '{MCT_NM}' & '{name}'")

df['result'] = results

df.to_csv('./res-info-result.csv', index=False)
