import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
import pandas as pd
import plotly.express as px
import streamlit as st

# Download required NLTK data automatically
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

st.set_page_config(
    page_title="WSD Engine (Unit III NLP)", page_icon="🧠", layout="wide"
)

st.title("🧠 Smart Word Sense Disambiguator")
st.caption(
    "JNTUH NLP Syllabus - Unit III: Semantic Parsing I & Word Sense Disambiguation"
)

st.markdown(
    """
### What is Word Sense Disambiguation?
In Natural Language Processing, many words are **polysemous** (they have multiple meanings). 
This app uses **Lesk's Algorithm** to compare the context of your sentence with dictionary definitions in **WordNet** to find the exact intended meaning!
"""
)

st.divider()

# Input UI
col1, col2 = st.columns([2, 1])

with col1:
    sentence = st.text_input(
        "Enter Context Sentence:",
        value="I went to the bank to deposit my money.",
        help="Type a sentence containing an ambiguous word.",
    )

with col2:
    target_word = st.text_input(
        "Target Word to Disambiguate:",
        value="bank",
        help="The word whose meaning you want to find.",
    )

if st.button("🚀 Disambiguate Sense", use_container_width=True):
    if not sentence or not target_word:
        st.error("Please provide both a sentence and a target word!")
    else:
        tokens = word_tokenize(sentence.lower())
        target_clean = target_word.strip().lower()

        # Run Lesk Algorithm
        best_sense = lesk(tokens, target_clean)

        # Get all synsets for comparison
        all_synsets = wn.synsets(target_clean)

        if not all_synsets:
            st.error(
                f"No definitions found in WordNet for the word '{target_word}'."
            )
        else:
            st.subheader("🎯 Disambiguation Result")

            if best_sense:
                st.success(
                    f"**Predicted Meaning (Synset):** `{best_sense.name()}`"
                )
                st.write(
                    f"**Definition:** {best_sense.definition().capitalize()}"
                )

                if best_sense.examples():
                    st.write(
                        f"**Usage Example:** *\"{best_sense.examples()[0]}\"*"
                    )
            else:
                st.warning(
                    "Lesk couldn't determine a clear sense from the given context."
                )

            st.divider()

            # Detailed Breakdowns
            st.subheader("📊 Candidate Senses & Overlap Analysis")

            stopwords = set(nltk.corpus.stopwords.words("english"))
            context_words = set(tokens) - stopwords - {target_clean}

            table_data = []
            overlap_scores = []

            for syn in all_synsets:
                # Calculate simple Lesk overlap manually for visualization
                def_tokens = set(word_tokenize(syn.definition().lower()))
                for ex in syn.examples():
                    def_tokens.update(word_tokenize(ex.lower()))

                def_words = def_tokens - stopwords
                overlap = len(context_words.intersection(def_words))

                table_data.append(
                    {
                        "Synset": syn.name(),
                        "POS": syn.pos(),
                        "Definition": syn.definition(),
                        "Context Overlap Count": overlap,
                    }
                )
                overlap_scores.append(
                    {"Synset": syn.name(), "Overlap": overlap}
                )

            df_senses = pd.DataFrame(table_data)

            # Bar plot of overlap scores
            df_scores = pd.DataFrame(overlap_scores)
            fig = px.bar(
                df_scores,
                x="Synset",
                y="Overlap",
                title="Context Overlap Score per Sense (Lesk Algorithm)",
                color="Overlap",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Data Table
            st.write("**All Candidate Meanings in WordNet:**")
            st.dataframe(df_senses, use_container_width=True)
