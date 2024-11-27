import os
from dotenv import load_dotenv
from pathlib import Path

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")


# Загрузим переменные среды из файла .env
load_dotenv()

# Настройка страницы
st.set_page_config(
    page_title="Playwise",
    page_icon="🎮",
    layout="wide"
)

# Определение путей до загружаемых данных
script_path = Path(__file__).resolve()
script_dir = script_path.parent

path_encoder = script_dir / 'data' / 'target_encoder.joblib'
path_scaler = script_dir / 'data' / 'standard_scaler.joblib'
path_catboost = script_dir / 'data' / 'catboost_first.cbm'
path_dashboard = script_dir / 'data' / 'data_for_dashboard.csv'
path_image = script_dir / 'data' / 'title.png'
path_image = os.path.abspath(path_image)

# Определим директорию с векторным хранилищем
persistent_directory = script_dir / 'data' / 'db' / 'chroma_db_with_metadata'
# Нормализуем путь для удобства
persistent_directory = os.path.abspath(persistent_directory)

# Инициализация состояния сеанса
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'top_game_ids' not in st.session_state:
    st.session_state.top_game_ids = []

# Добавляем инициализацию description_submitted
if 'description_submitted' not in st.session_state:
    st.session_state.description_submitted = False

# Загрузка датасета
@st.cache_resource
def load_dataset():
    df = pd.read_csv(path_dashboard)
    return df

# Загрузка предобработчиков
@st.cache_resource
def load_encoder():
    encoder = joblib.load(path_encoder)
    return encoder

@st.cache_resource
def load_scaler():
    scaler = joblib.load(path_scaler)
    return scaler

# Загрузка модели
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(path_catboost)
    return model

# Создаём необходимые объекты один раз и сохраняем их в состоянии сеанса
if 'initialized' not in st.session_state:
    # Загрузка датасета
    st.session_state.dataset = load_dataset()

    # Определим модель для эмбеддингов
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Загрузим существующее векторное хранилище и выбранную модель эмбеддингов
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings)

    # Создаем ретривер для поиска в векторном хранилище
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 50})
    st.session_state.retriever = retriever

    # Создаем нужную LLM модель
    llm = ChatOpenAI(model="gpt-4o")

    # Контекстуализация промта вопроса
    # Этот промт помогает ИИ понять, что следует переформулировать вопрос
    # и на основе истории чата сделать его отдельным вопросом
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    # Создание шаблон промта для контекстуализации вопроса
    # вариант с туториала
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Промт для ответа на вопрос
    qa_system_prompt = (
        "You are an experienced game developer and an expert in Game Design. "
        "Compare the user-entered description of the game and pieces of retrieved context. "
        "Provide feedback on the user's game idea, considering the similarities "
        "and differences with the listed games. Offer suggestions on how to make "
        "the game stand out or improve upon existing concepts. "
        "Try to give a short but informative answer."
        "\n\n"
        "{context}"
    )

    # Вариант с туториала 
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Создание history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # Создание цепочки для ответа на вопрос
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Создание РАГ цепочки, которая учитывает историю и цепочку ответа на вопрос
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain)

    # Загрузка предобработчиков и модели
    st.session_state.encoder = load_encoder()
    st.session_state.scaler = load_scaler()
    st.session_state.model = load_model()

    # Сохраняем объекты в состоянии сеанса
    st.session_state.llm = llm
    st.session_state.rag_chain = rag_chain
    st.session_state.initialized = True

# Вставка изображения на всю ширину страницы
st.image(path_image, use_container_width=True)

# Форма для ввода описания игры
with st.form(key='game_description_form'):
    user_input = st.text_area(
        "Enter a description of your game:",
        disabled=st.session_state.description_submitted
    )
    submit_button = st.form_submit_button(
        label='Send',
        disabled=st.session_state.description_submitted
    )

if not st.session_state.description_submitted and submit_button and user_input:
    # Добавляем вопрос пользователя в историю чата
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Выполняем поиск похожих игр
    relevant_docs = st.session_state.retriever.invoke(user_input)

    # Собираем уникальные идентификаторы игр из найденных документов
    unique_game_ids = []
    unique_relevant_docs = []
    for doc in relevant_docs:
        game_id = doc.metadata.get('game_id')
        if game_id not in unique_game_ids:
            unique_game_ids.append(game_id)
            unique_relevant_docs.append(doc)

    # Ограничиваем до топ-5 уникальных игр
    top_docs = unique_relevant_docs[:5]

    # Выводим информацию о топ-5 похожих игр
    st.subheader("Most similar games (Retrieval data from Chroma DB)")
    for i, doc in enumerate(top_docs, 1):
        st.markdown(f"""
        **Game #{i}**

        **Game ID:** {doc.metadata.get('game_id')}  
        **Title:** {doc.metadata.get('title', 'Not found')}  
        **Description:** {doc.page_content}
        """)
        # Добавляем горизонтальную линию между играми, кроме последней
        if i < len(top_docs):
            st.markdown("---")

    # Сохраняем топовые идентификаторы игр для дальнейшего использования
    st.session_state.top_game_ids = [
        doc.metadata.get('game_id') for doc in top_docs
        ]

    # Получаем ответ ассистента
    result = st.session_state.rag_chain.invoke(
        {"input": user_input, "chat_history": st.session_state.chat_history}
    )

    assistant_response = result['answer']

    # Добавление ответа ассистента в историю чата
    st.session_state.chat_history.append(
        SystemMessage(content=assistant_response)
        )

    # Инициализация сообщений чата
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
        )

    # Отображение ответа ассистента
    st.subheader("Playwise AI Assistant:")
    st.write(assistant_response)

    # Установка флага о вводе описания
    st.session_state.description_submitted = True

# Проверка наличия топ-3 игр
if st.session_state.top_game_ids:

    # Загрузка данных из состояния сеанса
    data_for_dashboard = st.session_state.dataset

    # Фильтрация данных по топ-3 играм
    top_game_ids = st.session_state.top_game_ids
    filtered_data = data_for_dashboard[
        data_for_dashboard['game_id'].isin(top_game_ids)
        ]

    # Удаление столбцов с нулевыми значениями
    filtered_data_non_zero = filtered_data.loc[:, (filtered_data != 0).any(axis=0)]
    filtered_data_non_zero.set_index('game_id', inplace=True)

    # Вывод датасета по похожим играм
    st.subheader("Most Similar Games Data")
    st.write(filtered_data_non_zero)

    # Создание колонок для дашборда
    col1, col2 = st.columns(2)

    # Вспомогательная функция для создания графиков с фиксированным размером
    def create_fixed_size_figure(figsize=(6, 4)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        return fig, ax

    # Вспомогательная функция для построения круговых диаграмм с легендой
    def plot_pie_with_legend(data, ax, title, autopct='%1.1f%%'):
        counts = data.value_counts()
        counts.plot(kind='pie', autopct=autopct, ax=ax, labels=None, startangle=90)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('')

        # Добавление легенды
        ax.legend(
            counts.index,
            title="Categories",
            loc='upper right',
            bbox_to_anchor=(1.3, 1),
            fontsize=8
        )

        # Изменение цвета подписей процентов на темно-серый
        for text in ax.texts:
            text.set_color('dimgray')  # Можно использовать 'darkgray'

        return ax

    with col1:
        # Распределение продаж по платформам
        platform_sales_columns = [
            col for col in filtered_data_non_zero.columns if 'sales' in col
            ]
        platform_sales_data = filtered_data_non_zero[platform_sales_columns]

        fig1, ax1 = create_fixed_size_figure()
        platform_sales_data.transpose().plot(kind='bar', stacked=True, ax=ax1)

        ax1.set_ylabel('Sales (mln)', fontsize=10)
        ax1.set_title('Distribution of sales by platforms', fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)

        # Удаление лишнего пространства вокруг графика
        fig1.tight_layout()

        st.pyplot(fig1)

        # Общие и среднегодовые продажи
        fig2, ax2 = create_fixed_size_figure()
        filtered_data_non_zero[
            ['global_all_platform', 'avg_sales_per_year']
            ].plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Sales (mln)', fontsize=10)
        ax2.set_title('Total and average annual sales', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

        # График распределения по жанрам
        # Распределение по жанрам
        fig_genres, ax_genres = create_fixed_size_figure()
        plot_pie_with_legend(
            filtered_data_non_zero['genres'], ax_genres, 'Distribution by genres'
            )
        st.pyplot(fig_genres)

    with col2:
        # Круговые диаграммы по категориальным переменным
        # Распределение по девелоперам
        fig_dev, ax_dev = create_fixed_size_figure()
        plot_pie_with_legend(
            filtered_data_non_zero['developer'], ax_dev, 'Distribution by developers'
            )
        st.pyplot(fig_dev)

        # Распределение по publisher
        fig_pub, ax_pub = create_fixed_size_figure()
        plot_pie_with_legend(
            filtered_data_non_zero['publisher'], ax_pub, 'Distribution by publishers'
            )
        st.pyplot(fig_pub)

        # Распределение по product_rating
        fig_rate, ax_rate = create_fixed_size_figure()
        plot_pie_with_legend(
            filtered_data_non_zero['product_rating'], ax_rate, 'Distribution by product rating'
            )
        st.pyplot(fig_rate)

    # Создание вкладок
    tab1, tab2 = st.tabs(["Playwise AI Assistant", "Sales Forecasting"])

    # Первая вкладка: Продолжение общения с LLM
    with tab1:
        st.subheader("Chat with Playwise AI Assistant")

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Контейнер для сообщений чата
        chat_container = st.container()

        # Поле ввода сообщения
        user_input = st.chat_input("Enter your message")

        # Если пользователь ввел сообщение
        if user_input:
            # Добавление сообщения пользователя
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
                )

            # Добавление в историю чата
            st.session_state.chat_history.append(
                HumanMessage(content=user_input)
                )

            # Получение ответа ассистента
            result = st.session_state.rag_chain.invoke(
                {"input": user_input, "chat_history": st.session_state.chat_history}
            )

            assistant_response = result['answer']

            # Добавление ответа ассистента в историю чата
            st.session_state.chat_history.append(
                SystemMessage(content=assistant_response)
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
                )

        # Отображение истории сообщений внутри контейнера
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    # Вторая вкладка: Предсказание продаж
    with tab2:
        st.header("Enter data to predict average annual sales")

        # Получение уникальных значений для категориальных признаков
        developer_options = sorted(
            st.session_state.dataset['developer'].dropna().unique()
            )
        publisher_options = sorted(
            st.session_state.dataset['publisher'].dropna().unique()
            )
        product_rating_options = sorted(
            st.session_state.dataset['product_rating'].dropna().unique()
            )
        genres_options = sorted(
            st.session_state.dataset['genres'].dropna().unique()
            )

        with st.form(key='prediction_form'):
            # Поля ввода
            meta_score_input = st.number_input(
                'Meta Score (0-100, step=5)',
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=5.0)
            user_review_input = st.number_input(
                'User Review (0-10, step=0.5)',
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.5)
            release_date_input = st.date_input('Release Date')

            # Выпадающие списки
            developer_input = st.selectbox('Developer', developer_options)
            publisher_input = st.selectbox('Publisher', publisher_options)
            genres_input = st.selectbox('Genre', genres_options)
            product_rating_input = st.selectbox(
                'Product Rating', product_rating_options
                )

            # Кнопка предсказания
            predict_button = st.form_submit_button(
                label='Predict average annual sales'
                )

        if predict_button:
            # Создание DataFrame из введённых данных
            input_data = pd.DataFrame({
                'meta_score': [meta_score_input],
                'user_review': [user_review_input],
                'release_date': [release_date_input],
                'developer': [developer_input],
                'publisher': [publisher_input],
                'genres': [genres_input],
                'product_rating': [product_rating_input]
            })

            # Преобразование даты релиза
            input_data['release_date'] = pd.to_datetime(
                input_data['release_date']
                )

            # Кодирование категориальных признаков
            columns_to_encode = [
                'release_date', 'developer', 'publisher', 'genres', 'product_rating'
                ]
            input_encoded = st.session_state.encoder.transform(
                input_data[columns_to_encode]
                )
            input_encoded = pd.DataFrame(
                input_encoded, columns=columns_to_encode
                )

            # Объединение с числовыми признаками
            numeric_features = ['meta_score', 'user_review']
            input_numeric = input_data[numeric_features].reset_index(drop=True)
            input_preprocessed = pd.concat(
                [input_encoded.reset_index(drop=True), input_numeric], axis=1
                )

            # Упорядочивание колонок
            feature_names = st.session_state.scaler.feature_names_in_
            input_preprocessed = input_preprocessed[feature_names]

            # Масштабирование данных
            input_scaled = st.session_state.scaler.transform(input_preprocessed)

            # Получение предсказания
            prediction = st.session_state.model.predict(input_scaled)

            # Отображение результата
            st.header(
                f'**Projected average annual sales:** {prediction[0]:.4f} million per year'
                )


            # Раздел Disclaimer
            st.markdown("### Disclaimer:")

            disclaimer_text = """
            - **The model** identifies patterns with an accuracy of 71% (R² score).
            - **Increasing the advertising budget** is likely to lead to an annual increase in the number of copies sold.
            - **Our dataset** does not contain information on advertising expenditures, as this is often considered confidential.
            - **The absence of this data** may affect the accuracy of the identified patterns.
            - **We are working** on expanding the dataset's feature set to enhance the precision of the forecasts provided.
            """

            st.markdown(disclaimer_text)
