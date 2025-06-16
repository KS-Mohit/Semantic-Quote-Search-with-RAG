
import streamlit as st
import pandas as pd
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Quotes RAG Application",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model and index loading
@st.cache_resource
def load_model_and_index():
    """Load the fine-tuned model and vector index"""
    try:
        # Load model
        model = SentenceTransformer("./fine_tuned_quotes_model")
        
        # Load FAISS index
        index = faiss.read_index("quotes_index.faiss")
        
        # Load data
        with open("quotes_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        return model, index, data['df']
    except Exception as e:
        st.error(f"Error loading model and index: {e}")
        return None, None, None

class StreamlitRAGPipeline:
    def __init__(self, model, index, df):
        self.model = model
        self.index = index
        self.df = df
    
    def search_quotes(self, query, top_k=5):
        """Search for quotes using the RAG pipeline"""
        if self.index is None or self.model is None:
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            # Get results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:
                    row = self.df.iloc[idx]
                    results.append({
                        'rank': i + 1,
                        'quote': row.get('quote', row.get('text', '')),
                        'author': row.get('author', row.get('title', '')),
                        'tags': row.get('tags', ''),
                        'similarity_score': float(score),
                        'index': int(idx)
                    })
            
            return results
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def generate_response(self, query, results):
        """Generate a structured response"""
        if not results:
            return {
                'query': query,
                'summary': 'No relevant quotes found for your query.',
                'quotes': [],
                'total_found': 0,
                'authors': [],
                'themes': []
            }
        
        # Extract information
        authors = list(set([r['author'] for r in results if r['author'] != 'Unknown']))
        tags = []
        for r in results:
            if r['tags']:
                tags.extend(str(r['tags']).split(','))
        
        unique_tags = list(set([tag.strip() for tag in tags if tag.strip()]))[:5]
        
        # Generate summary
        summary = f"Found {len(results)} relevant quotes"
        if authors:
            summary += f" from authors including {', '.join(authors[:3])}"
        if unique_tags:
            summary += f" related to themes like {', '.join(unique_tags[:3])}"
        
        return {
            'query': query,
            'summary': summary,
            'quotes': results,
            'total_found': len(results),
            'authors': authors,
            'themes': unique_tags
        }

def main():
    # Header
    st.title("📚 Quotes RAG Application")
    st.markdown("**Retrieval Augmented Generation for English Quotes**")
    st.markdown("Search for quotes using natural language queries!")
    
    # Load model and index
    with st.spinner("Loading model and index..."):
        model, index, df = load_model_and_index()
    
    if model is None or index is None or df is None:
        st.error("Failed to load model and index. Please ensure all files are present.")
        return
    
    # Initialize RAG pipeline
    rag_pipeline = StreamlitRAGPipeline(model, index, df)
    
    # Sidebar
    st.sidebar.header("Search Settings")
    top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)
    show_scores = st.sidebar.checkbox("Show similarity scores", value=True)
    
    # Example queries
    st.sidebar.header("Example Queries")
    example_queries = [
        "quotes about hope by Oscar Wilde",
        "inspirational quotes about success",
        "quotes about love by Shakespeare",
        "motivational quotes for difficult times",
        "funny quotes about life",
        "quotes about courage by women authors"
    ]
    
    for example in example_queries:
        if st.sidebar.button(f"📝 {example}", key=example):
            st.session_state.query = example
    
    # Main search interface
    st.header("🔍 Search Quotes")
    
    # Query input
    query = st.text_input(
        "Enter your query:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., 'Show me quotes about courage by women authors'",
        key="main_query"
    )
    
    # Search button
    if st.button("Search", type="primary") or query:
        if query.strip():
            with st.spinner("Searching quotes..."):
                # Perform search
                results = rag_pipeline.search_quotes(query, top_k)
                response = rag_pipeline.generate_response(query, results)
            
            # Display results
            st.header("📋 Results")
            
            # Summary
            st.info(f"**Summary:** {response['summary']}")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Found", response['total_found'])
            with col2:
                st.metric("Authors", len(response['authors']))
            with col3:
                st.metric("Themes", len(response['themes']))
            
            # Display quotes
            if response['quotes']:
                st.subheader("📖 Quotes")
                
                for i, quote_data in enumerate(response['quotes']):
                    with st.expander(f"Quote {i+1}: {quote_data['author']}", expanded=i<3):
                        st.markdown(f'**Quote:** *"{quote_data["quote"]}"*')
                        st.markdown(f"**Author:** {quote_data['author']}")
                        if quote_data['tags']:
                            st.markdown(f"**Tags:** {quote_data['tags']}")
                        if show_scores:
                            st.markdown(f"**Similarity Score:** {quote_data['similarity_score']:.4f}")
                
                # JSON Response
                st.subheader("🔧 JSON Response")
                with st.expander("View structured JSON response"):
                    st.json(response)
            
            # Additional insights
            if response['authors'] or response['themes']:
                st.subheader("📊 Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if response['authors']:
                        st.write("**Top Authors:**")
                        for author in response['authors'][:5]:
                            st.write(f"• {author}")
                
                with col2:
                    if response['themes']:
                        st.write("**Related Themes:**")
                        for theme in response['themes'][:5]:
                            st.write(f"• {theme}")
        
        else:
            st.warning("Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>📚 Quotes RAG Application | Built with Streamlit & Sentence Transformers</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
