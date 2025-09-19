# app.py
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from openai import OpenAI

# Config
st.set_page_config(page_title="Chatbot: Tabla de Clientes", layout="wide")
st.title("Chatbot: Preguntas sobre la tabla de clientes")

# Cargar datos (desde GitHub)
DATA_URL = "https://raw.githubusercontent.com/eugeniomorocho/theFutureOfDataScience/main/mall_customers.csv"
@st.cache_data
def load_data(url=DATA_URL):
    df = pd.read_csv(url)
    return df

df = load_data()

st.subheader("Tabla de datos")
st.dataframe(df)

st.markdown("---")
st.subheader("Haz una pregunta sobre la tabla")

# Entrada de usuario
user_question = st.text_input("Escribe tu pregunta (sobre la tabla):", key="question_input")
ask_button = st.button("Consultar")

# Inicializar cliente OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    st.warning("No se encontró la variable OPENAI_API_KEY. Define la clave en las variables de entorno (.env) antes de usar la API.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Heurística simple para detectar si la pregunta parece no relacionada:
def is_related_to_table(question: str, df: pd.DataFrame) -> bool:
    # Si contiene columnas o palabras clave relacionadas con la tabla, la consideramos relacionada.
    q = question.lower()
    columnas = [c.lower() for c in df.columns]
    # si alguna columna está en la pregunta, o aparece la palabra cliente, ingresos, age, spending, gender
    keywords = columnas + ['cliente', 'clientes', 'income', 'ingreso', 'spending', 'gasto', 'age', 'edad', 'gender', 'sexo', 'id']
    matches = sum(1 for k in keywords if k in q)
    # si hay al menos una coincidencia, asumimos relacionada; este chequeo evita preguntas generales.
    return matches >= 1

# Convertir (opcional) la tabla a CSV string — cuidado con tokens: si tu tabla es grande
def df_to_limited_csv(df: pd.DataFrame, max_rows=200) -> str:
    return df.head(max_rows).to_csv(index=False)

# Cuando consultan:
if ask_button and user_question:
    if not is_related_to_table(user_question, df):
        st.info("Lo siento — sólo puedo responder preguntas relacionadas con la tabla mostrada.")
    else:
        if client is None:
            st.error("OpenAI no está configurado. Añade OPENAI_API_KEY en variables de entorno.")
        else:
            # Preparar prompt estricto: instrucción que obliga a usar sólo la tabla
            system_prompt = (
                "Eres un asistente cuya única fuente de información permitida es la tabla CSV proporcionada. "
                "RESPONDE SÓLO con información que esté directamente en esa tabla. "
                "Si la pregunta no se puede responder con la tabla, responde exactamente: "
                "'Lo siento, sólo puedo responder preguntas sobre la tabla mostrada.' "
                "No inventes valores ni supongas datos no presentes."
            )

            tabla_csv = df_to_limited_csv(df, max_rows=200)

            user_message = f"""
            Aquí está la tabla (CSV):
            {tabla_csv}

            Pregunta: {user_question}

            Responde de forma concisa usando sólo la tabla.
            """

            with st.spinner("Consultando modelo..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # ajustar por disponibilidad/permiso de tu cuenta
                        messages=[
                            {"role":"system", "content": system_prompt},
                            {"role":"user", "content": user_message}
                        ],
                        max_tokens=300,
                    )
                    answer = response.choices[0].message.content.strip()
                except Exception as e:
                    st.error(f"Error en la llamada a OpenAI: {e}")
                    answer = None

            if answer:
                st.markdown("**Respuesta del chatbot:**")
                st.write(answer)
                # Mostrar la fuente (fila(s) relevantes) — opcional: simple heurística
                st.markdown("---")
                st.markdown("**Filas de la tabla relacionadas (búsqueda heurística):**")
                # Buscamos coincidencias de palabras clave en filas para dar contexto
                q_tokens = [t for t in user_question.lower().split() if len(t) > 2]
                mask = pd.Series(False, index=df.index)
                for tok in q_tokens:
                    mask = mask | df.astype(str).apply(lambda col: col.str.lower().str.contains(tok, na=False)).any(axis=1)
                related = df[mask]
                if related.empty:
                    st.write("No se encontraron filas coincidentes por heurística.")
                else:
                    st.dataframe(related)
                st.markdown("*Nota: La búsqueda de filas relacionadas es una heurística simple y puede no ser perfecta.*")