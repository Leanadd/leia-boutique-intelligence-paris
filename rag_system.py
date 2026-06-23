"""
LÉIA RAG System
Manages knowledge base loading, embeddings, and retrieval
"""

import os
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class LEIAKnowledgeBase:
    """
    Manages LÉIA's knowledge base for RAG system
    """
    
    def __init__(self, knowledge_base_path="knowledge_base"):
        self.kb_path = Path(knowledge_base_path)
        self.documents = []
        self.vectorstore = None
        self.embeddings = None
        
    def load_text_files(self):
        """Load all text files from knowledge base"""
        print("📚 Loading text files...")
        
        text_files = [
            "brand_story.txt",
            "after_sales_policy.txt",
            "care_instructions.txt",
            "boutique_guidelines.txt",
            "boutique_innovation.txt"
        ]
        
        for filename in text_files:
            filepath = self.kb_path / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "type": "policy" if "policy" in filename else "guide"
                        }
                    )
                    self.documents.append(doc)
                    print(f"   ✅ Loaded {filename}")
                except Exception as e:
                    print(f"   ❌ Error loading {filename}: {e}")
            else:
                print(f"   ⚠️  File not found: {filename}")
    
    def load_csv_files(self):
        """Load CSV files and convert to documents"""
        print("📊 Loading CSV files...")
        
        # Load products
        products_file = self.kb_path / "leia_products.csv"
        if products_file.exists():
            try:
                # Skip comment lines that start with #
                df = pd.read_csv(products_file, comment='#', on_bad_lines='skip')
        
                for _, row in df.iterrows():
                    content = f"""
Product: {row['name']}
Collection: {row['collection']}
Category: {row['category']}
Price: ${row['price_usd']:,}
Materials: {row['materials']}
Gemstones: {row['gemstones']}
Story: {row['story']}
Craftsmanship: {row['craftsmanship']}
Care: {row['care_instructions']}
                    """.strip()
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "products",
                            "product_id": row['product_id'],
                            "collection": row['collection'],
                            "price": row['price_usd']
                        }
                    )
                    self.documents.append(doc)
                
                print(f"   ✅ Loaded {len(df)} products")
            except Exception as e:
                print(f"   ❌ Error loading products: {e}")
        
        # Load client profiles
        clients_file = self.kb_path / "client_profiles.csv"
        if clients_file.exists():
            try:
                # encoding for special characters
                df = pd.read_csv(clients_file, encoding='utf-8')
                df = df[~df['client_id'].astype(str).str.startswith('#')]
                df = df.dropna(subset=['client_id'])
                df["full_name"] = df["first_name"] + " " + df["last_name"]
                
                for _, row in df.iterrows():
                    content = f"""
Client: {row['first_name']} {row['last_name']}
Pronouns: {row['preferred_pronouns']}
City: {row['city']}
VIP Tier: {row['vip_tier']}
Total Spent: ${row['total_spent_usd']:,}
Purchases: {row['purchase_count']}
Last Visit: {row['last_visit_date']}
Birthday: {row['birthdate']}
Preferred Collections: {row['preferred_collections']}
Style: {row['style_profile']}
Budget Range: {row['budget_range']}
Notable Purchases: {row['notable_purchases']}
Notes: {row['notes']}
                    """.strip()
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "clients",
                            "client_id": row['client_id'],
                            "city": row['city']
                        }
                    )
                    self.documents.append(doc)
                
                print(f"   ✅ Loaded {len(df)} client profiles")
            except Exception as e:
                print(f"   ❌ Error loading clients: {e}")
        
        # Load purchase history
        purchases_file = self.kb_path / "purchase_history.csv"
        if purchases_file.exists():
            try:
                df = pd.read_csv(purchases_file)
                # Remove comment rows
                df = df[~df['transaction_id'].astype(str).str.startswith('#')]
                
                for _, row in df.iterrows():
                    content = f"""
Transaction: {row['transaction_id']}
Client: {row['client_id']}
Date: {row['date']}
Product: {row['product_name']} ({row['collection']})
Price: ${row['price_usd']:,}
Occasion: {row['occasion']}
Payment: {row['payment_method']}
Boutique: {row['boutique_city']}
Advisor: {row['advisor_name']}
Notes: {row['advisor_notes']}
                    """.strip()
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "purchases",
                            "transaction_id": row['transaction_id'],
                            "client_id": row['client_id']
                        }
                    )
                    self.documents.append(doc)
                
                print(f"   ✅ Loaded {len(df)} purchase records")
            except Exception as e:
                print(f"   ❌ Error loading purchases: {e}")
    
    def chunk_documents(self):
        """Split large documents into smaller chunks"""
        print("✂️  Chunking documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Size of each chunk
            chunk_overlap=200,  # Overlap between chunks
            length_function=len,
        )
        
        chunked_docs = text_splitter.split_documents(self.documents)
        self.documents = chunked_docs
        print(f"   ✅ Created {len(chunked_docs)} chunks")
    
    def create_embeddings(self):
        """Create embeddings and vector store"""
        print("🧠 Creating embeddings (this may take a minute)...")
        
        # Use HuggingFace embeddings (free, runs locally)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"  # Meilleur modèle
            )
        
        # without persist_directory 
    self.vectorstore = Chroma.from_documents(
        documents=self.documents,
        embedding=self.embeddings,
    )
        
        print("   ✅ Embeddings created and stored")
    
    def initialize(self):
        """Initialize the entire knowledge base"""
        print("\n" + "="*50)
        print("🚀 Initializing LÉIA Knowledge Base")
        print("="*50 + "\n")
        
        self.load_text_files()
        self.load_csv_files()
        
        print(f"\n📦 Total documents loaded: {len(self.documents)}")
        
        if len(self.documents) == 0:
            print("❌ No documents loaded! Check your knowledge_base folder.")
            return False
        
        self.chunk_documents()
        self.create_embeddings()
        
        print("\n" + "="*50)
        print("✅ Knowledge Base Ready!")
        print("="*50 + "\n")
        
        return True
    
    def query(self, question, k=5):
        """
        Query the knowledge base
        
        Args:
            question: User's question
            k: Number of relevant documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            print("❌ Vector store not initialized!")
            return []
        
        # Search for relevant documents
        results = self.vectorstore.similarity_search(question, k=k)
        return results


# Test function
if __name__ == "__main__":
    # Test the knowledge base
    kb = LEIAKnowledgeBase()
    
    if kb.initialize():
        print("\n🧪 Testing query...")
        results = kb.query("Tell me about the Eclipse collection")
        
        print(f"\nFound {len(results)} relevant documents:")
        for i, doc in enumerate(results[:2], 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
