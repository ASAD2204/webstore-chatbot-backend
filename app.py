# ================================================================
# WEBSTORE AI CHATBOT BACKEND - COMPREHENSIVE E-COMMERCE AI SYSTEM
# ================================================================
# 
# SYSTEM OVERVIEW:
# This is the main backend server for WebStore's AI-powered customer service chatbot.
# It combines multiple AI technologies to provide intelligent customer support,
# business analytics, and visual product search capabilities.
#
# ARCHITECTURE COMPONENTS:
# 1. Flask Web Server - RESTful API endpoints for frontend communication
# 2. Google Gemini AI - Natural language understanding and response generation
# 3. MySQL Database - Product catalog, orders, customers, and business data
# 4. ChromaDB Vector Database - Semantic search and knowledge retrieval
# 5. BLIP Image AI - Visual product analysis and similarity matching
# 6. Sentence Transformers - Text embeddings for semantic search
# 7. Telegram Integration - Mobile chatbot interface
#
# CORE FUNCTIONALITIES:
# - Customer Chat: Answer questions, track orders, recommend products
# - Admin Analytics: Real-time business intelligence and reporting
# - Image Search: Visual product matching using AI image analysis
# - Order Tracking: Real-time shipping status with tracking numbers
# - Product Search: Intelligent product discovery with filters
# - Knowledge Base: RAG (Retrieval Augmented Generation) for accurate responses
#
# DATA FLOW:
# Frontend → Flask API → Intent Analysis → Knowledge Search → AI Response → Frontend
#                    ↓
#              MySQL/ChromaDB Query → Context Building → Gemini AI → Formatted Response
#
# HOW THE CHATBOT WORKS:
# 1. CUSTOMER INTERACTION: User sends message through web interface or Telegram
# 2. INTENT ANALYSIS: AI analyzes message to understand what user wants
# 3. CONTEXT GATHERING: System gathers relevant info from database and knowledge base
# 4. AI PROCESSING: Google Gemini generates intelligent response using gathered context
# 5. RESPONSE DELIVERY: Formatted response sent back to user with actionable information
#
# ADMIN FEATURES:
# - Real-time business analytics and performance reporting
# - Sales trends and customer behavior analysis
# - Inventory management with low stock alerts
# - Customer service inquiry tracking and response
#
# IMAGE SEARCH FEATURES:
# - Upload image to find similar products
# - AI-powered category detection from images
# - Visual similarity matching using deep learning
# - Product recommendations based on visual features
#
# DEPLOYMENT REQUIREMENTS:
# - Python 3.8+ with required packages (see requirements.txt)
# - MySQL database with eshop schema
# - Google Gemini AI API key
# - Sufficient disk space for ChromaDB vector storage
# - Optional: Telegram Bot Token for mobile integration
# ================================================================

# ================================================================
# IMPORT DEPENDENCIES - ORGANIZED BY CATEGORY
# ================================================================

# ================================================================
# 1. CORE WEB FRAMEWORK IMPORTS
# ================================================================
from flask import Flask, request, jsonify, session  # Web server and HTTP handling
from flask_cors import CORS                          # Cross-origin resource sharing

# ================================================================
# 2. DATABASE AND DATA PROCESSING IMPORTS
# ================================================================
import mysql.connector                       # MySQL database connectivity for e-commerce data
import json                                  # JSON data parsing and serialization
import os                                   # Operating system interface
import re                                   # Regular expression pattern matching for intent analysis
import uuid                                 # Unique identifier generation for sessions
import logging                              # Application logging and debugging
from datetime import datetime               # Date and time operations for timestamps
from typing import List, Dict, Optional, Any # Type hints for better code quality

# ================================================================
# 3. AI AND MACHINE LEARNING IMPORTS
# ================================================================
import google.generativeai as genai         # Google Gemini AI for natural language processing
import chromadb                             # Vector database for semantic search and RAG
from sentence_transformers import SentenceTransformer  # Text embeddings for similarity
from transformers import BlipProcessor, BlipForConditionalGeneration  # Image AI models
import torch                                # PyTorch deep learning framework
import numpy as np                          # Numerical computing operations
from sklearn.metrics.pairwise import cosine_similarity  # Similarity calculations

# ================================================================
# 4. IMAGE PROCESSING AND ENCODING IMPORTS
# ================================================================
import base64                               # Base64 encoding/decoding for images
from PIL import Image                       # Python Imaging Library for image manipulation
import io                                   # Input/output stream operations
import requests                             # HTTP requests (if needed for external APIs)

# ================================================================
# 5. CONFIGURATION AND ENVIRONMENT IMPORTS
# ================================================================
from dotenv import load_dotenv              # Environment variable management

# Load environment variables from .env file (API keys, database credentials)
load_dotenv()

# Import custom Telegram bot integration module
from telegram_integration import register_telegram_routes

# ================================================================
# 6. LOGGING CONFIGURATION
# ================================================================
# Configure comprehensive logging for debugging, monitoring, and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# 7. FLASK APPLICATION INITIALIZATION
# ================================================================
# Create Flask app instance with security configurations
app = Flask(__name__)
app.secret_key = 'webstore-chatbot-secret-key-2025'  # Change in production

# Configure CORS for cross-origin requests (frontend communication)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'], supports_credentials=True)

# ================================================================
# 8. CONFIGURATION CLASS - CENTRALIZED SETTINGS
# ================================================================
class Config:
    """
    Centralized configuration management for all system components.
    This class contains all API keys, database credentials, and system settings.
    All sensitive values are loaded from environment variables.
    """
    
    # Google Gemini AI Configuration
    # Used for natural language understanding and response generation
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-api-key-here')  # Load from environment
    
    # MySQL Database Configuration
    # Main database storing products, orders, customers, and business data
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),        # Database server location
        'user': os.getenv('DB_USER', 'root'),            # Database username
        'password': os.getenv('DB_PASSWORD', ''),        # Database password from environment
        'database': os.getenv('DB_NAME', 'eshop'),       # Database name containing all e-commerce tables
        'autocommit': True                                # Automatic transaction commits
    }
    
    # Vector Database Configuration
    # ChromaDB settings for semantic search and AI knowledge base
    VECTOR_DB_PATH = "./vector_db"              # Local storage path for vector embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # Sentence transformer model for text embeddings

# ================================================================
# 9. AI MODELS INITIALIZATION
# ================================================================
# Initialize Google Gemini AI for natural language processing
genai.configure(api_key=Config.GEMINI_API_KEY)

# Use the working models from diagnostic - prioritize Flash models for better performance and quota
WORKING_GEMINI_MODELS = [
    'models/gemini-2.5-flash',      # ✅ BEST: Latest stable Flash model (confirmed working)
    'models/gemini-2.0-flash',      # ✅ Alternative Flash model (confirmed working)
    'models/gemini-2.0-flash-001',  # ✅ Stable Flash model (confirmed working)
    'models/gemini-2.5-pro',        # ✅ Pro model (confirmed working)
]

gemini_model = None
for model_name in WORKING_GEMINI_MODELS:
    try:
        gemini_model = genai.GenerativeModel(model_name)
        # Test the model with a simple prompt
        test_response = gemini_model.generate_content("Hello")
        print(f"🎉 Successfully initialized Gemini model: {model_name}")
        print(f"   Test response: {test_response.text.strip()}")
        break
    except Exception as e:
        print(f"❌ Failed to initialize {model_name}: {str(e)[:100]}...")
        continue

if gemini_model is None:
    print("🚨 CRITICAL: No Gemini model could be initialized!")
    print("🔧 Falling back to basic text responses...")
    # Create a dummy model for graceful degradation
    class DummyModel:
        def generate_content(self, prompt):
            class DummyResponse:
                text = "I'm currently experiencing technical difficulties. Please contact our support team for assistance."
            return DummyResponse()
    gemini_model = DummyModel()

# Initialize sentence transformer for creating text embeddings
# This model converts text into numerical vectors for similarity matching
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

# ================================================================
# 10. IMAGE PROCESSING MODELS INITIALIZATION
# ================================================================
# Initialize BLIP (Bootstrapped Language-Image Pre-training) for image analysis
# These models can understand and describe images for product matching
try:
    # Load image captioning processor and model
    image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    logger.info("✅ Image processing models loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load image processing models: {e}")
    image_model = None

# ================================================================
# 11. DATABASE CONNECTIONS INITIALIZATION
# ================================================================

# MySQL Database Connection
# Connects to the main e-commerce database containing all business data
try:
    mysql_db = mysql.connector.connect(**Config.DB_CONFIG)
    logger.info("✅ MySQL connection established")
except mysql.connector.Error as err:
    logger.error(f"❌ MySQL connection failed: {err}")
    mysql_db = None

# ChromaDB Vector Database Initialization
# Used for semantic search and AI knowledge base storage
try:
    # Create persistent client for vector database
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
    
    # Create or retrieve existing knowledge collection
    try:
        knowledge_collection = chroma_client.get_collection("webstore_knowledge")
        logger.info("✅ Loaded existing ChromaDB collection")
    except:
        knowledge_collection = chroma_client.create_collection("webstore_knowledge")
        logger.info("✅ Created new ChromaDB collection")
        
except Exception as e:
    logger.error(f"❌ ChromaDB initialization failed: {e}")
    knowledge_collection = None

# ================================================================
# 12. WEBSTORE KNOWLEDGE BASE CLASS
# ================================================================
class WebStoreKnowledgeBase:
    """
    Comprehensive knowledge management system for the WebStore chatbot.
    
    This class handles:
    - Website information and features storage
    - Product catalog embedding and search
    - Business policies and procedures
    - Customer service knowledge base
    - Vector-based semantic search capabilities
    """
    
    def __init__(self, mysql_connection, chroma_collection, embedding_model):
        """
        Initialize the knowledge base with database connections and AI models.
        
        Args:
            mysql_connection: MySQL database connection for product data
            chroma_collection: ChromaDB collection for vector storage
            embedding_model: Sentence transformer for text embeddings
        """
        self.db = mysql_connection
        self.vector_db = chroma_collection
        self.embedder = embedding_model
        
        # Website features and capabilities information
        # This data helps the chatbot understand what the website offers
        self.website_features = {
            # Page-specific information for customer guidance
            "pages": {
                "/": "Homepage with hero section, featured products, and special offers",
                "/shop": "Product catalog with filtering, search, and sorting options",
                "/cart": "Shopping cart management with quantity updates and total calculation",
                "/checkout": "Secure checkout process with payment and shipping options",
                "/contact": "Contact form for customer inquiries and support",
                "/about": "About page with company information and story",
                "/login": "User login and registration system",
                "/user": "User account dashboard with order history and profile management",
                "/wishlist": "Product wishlist for saving favorite items",
                "/track-order": "Order tracking system with shipping updates",
                "/admin": "Admin panel with real-time statistics and store management",
                "/blog": "Blog section with articles and company updates"
            },
            "features": [
                "User registration and secure login system",
                "Product catalog with advanced search and filters",
                "Shopping cart and wishlist functionality", 
                "Secure checkout with multiple payment options",
                "Order tracking and management system",
                "Admin panel with real-time business analytics",
                "Contact form and customer support system",
                "Newsletter subscription for updates and offers",
                "Discount codes and promotional campaigns",
                "Mobile-responsive design for all devices",
                "Customer reviews and product ratings",
                "Inventory management with low stock alerts"
            ],
            "policies": {
                "shipping": "Free shipping on orders over $100. Standard shipping: 3-5 business days ($5). Express shipping: 1-2 business days ($15).",
                "returns": "30-day return policy. Items must be unused and in original packaging. Refunds issued to original payment method.",
                "payment": "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay.",
                "privacy": "We protect your personal information and comply with data protection regulations.",
                "support": "24/7 customer support available via chat, email, or phone."
            }
        }
    
    def extract_database_content(self) -> Dict[str, Any]:
        """Extract all relevant content from MySQL database"""
        if not self.db:
            return {}
            
        try:
            cursor = self.db.cursor(dictionary=True)
            content = {}
            
            # Extract products with inventory
            cursor.execute("""
                SELECT p.*, i.stock_quantity 
                FROM product p 
                LEFT JOIN inventory i ON p.product_id = i.product_id
            """)
            content['products'] = cursor.fetchall()
            
            # Extract product categories
            cursor.execute("SELECT DISTINCT product_type FROM product WHERE product_type IS NOT NULL")
            content['categories'] = [row['product_type'] for row in cursor.fetchall()]
            
            # Extract discount information
            cursor.execute("""
                SELECT * FROM discount 
                WHERE end_date >= NOW() OR end_date IS NULL
            """)
            content['discounts'] = cursor.fetchall()
            
            # Extract customer service messages for FAQ insights
            cursor.execute("""
                SELECT subject, message FROM messages 
                ORDER BY created_at DESC 
                LIMIT 100
            """)
            content['customer_inquiries'] = cursor.fetchall()
            
            cursor.close()
            return content
            
        except mysql.connector.Error as e:
            logger.error(f"Database content extraction failed: {e}")
            return {}
    
    def build_knowledge_embeddings(self):
        """Build vector embeddings for all knowledge"""
        if not self.vector_db:
            logger.error("Vector database not available")
            return
            
        logger.info("Building knowledge embeddings...")
        
        # Clear existing collection
        try:
            self.vector_db.delete(where={})
        except:
            pass
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        # 1. Embed website features and pages
        for page, description in self.website_features["pages"].items():
            text = f"Page {page}: {description}"
            embedding = self.embedder.encode(text)
            
            documents.append(text)
            embeddings.append(embedding.tolist())
            metadatas.append({"type": "page", "page": page})
            ids.append(f"page_{page.replace('/', '_')}")
        
        # 2. Embed features
        for i, feature in enumerate(self.website_features["features"]):
            text = f"Website feature: {feature}"
            embedding = self.embedder.encode(text)
            
            documents.append(text)
            embeddings.append(embedding.tolist())
            metadatas.append({"type": "feature", "feature_id": i})
            ids.append(f"feature_{i}")
        
        # 3. Embed policies
        for policy_type, policy_text in self.website_features["policies"].items():
            text = f"{policy_type.title()} Policy: {policy_text}"
            embedding = self.embedder.encode(text)
            
            documents.append(text)
            embeddings.append(embedding.tolist())
            metadatas.append({"type": "policy", "policy_type": policy_type})
            ids.append(f"policy_{policy_type}")
        
        # 4. Embed database content
        db_content = self.extract_database_content()
        
        # Embed products
        for product in db_content.get('products', []):
            text = f"Product: {product.get('product_name', '')} - {product.get('product_discription', '')} - Category: {product.get('product_type', '')} - Price: ${product.get('product_price', 0)} - Stock: {product.get('stock_quantity', 0)}"
            embedding = self.embedder.encode(text)
            
            documents.append(text)
            embeddings.append(embedding.tolist())
            metadatas.append({
                "type": "product", 
                "product_id": product.get('product_id'),
                "category": product.get('product_type'),
                "price": float(product.get('product_price', 0))
            })
            ids.append(f"product_{product.get('product_id')}")
        
        # Embed categories
        for category in db_content.get('categories', []):
            text = f"Product category: {category}"
            embedding = self.embedder.encode(text)
            
            documents.append(text)
            embeddings.append(embedding.tolist())
            metadatas.append({"type": "category", "category": category})
            ids.append(f"category_{category.lower().replace(' ', '_')}")
        
        # Add to vector database in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.vector_db.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        logger.info(f"Added {len(documents)} documents to vector database")
    
    def search_knowledge(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search knowledge base using vector similarity"""
        if not self.vector_db:
            return []
        
        try:
            query_embedding = self.embedder.encode(query)
            results = self.vector_db.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            knowledge_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    knowledge_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
            
            return knowledge_results
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []

# ================================================================
# CHAT HISTORY MANAGER - DATABASE PERSISTENCE
# ================================================================
class ChatHistoryManager:
    """
    Manages persistent chat history using MySQL database storage.
    
    This class handles:
    - Saving conversations to database
    - Retrieving chat history for sessions
    - Session management and analytics
    - Performance tracking and optimization
    """
    
    def __init__(self, mysql_connection):
        """
        Initialize chat history manager with database connection.
        
        Args:
            mysql_connection: MySQL database connection
        """
        self.db = mysql_connection
    
    def save_message(self, session_id: str, sender: str, message: str, 
                    intent: str = None, confidence: float = None, 
                    user_email: str = None, current_page: str = None,
                    response_time_ms: int = None) -> bool:
        """
        Save a chat message to the database.
        
        Args:
            session_id: Unique session identifier
            sender: 'user', 'bot', 'admin', or 'admin_bot'
            message: The message content
            intent: Detected intent (optional)
            confidence: AI confidence score (optional)
            user_email: Customer email if logged in (optional)
            current_page: Current website page (optional)
            response_time_ms: Response time in milliseconds (optional)
            
        Returns:
            Boolean indicating success/failure
        """
        if not self.db:
            logger.warning("Database not available for chat history storage")
            return False
            
        try:
            cursor = self.db.cursor()
            
            # Insert message into chat_conversations table
            query = """
                INSERT INTO chat_conversations 
                (session_id, user_email, sender, message, intent, confidence, 
                 current_page, response_time_ms) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                session_id, user_email, sender, message, 
                intent, confidence, current_page, response_time_ms
            ))
            
            # Update or create session record
            self._update_session_info(session_id, user_email, sender)
            
            cursor.close()
            logger.info(f"💾 Chat message saved: {session_id} - {sender}")
            return True
            
        except mysql.connector.Error as e:
            logger.error(f"❌ Failed to save chat message: {e}")
            return False
    
    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """
        Retrieve chat history for a specific session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries ordered by timestamp
        """
        if not self.db:
            logger.warning("Database not available for chat history retrieval")
            return []
            
        try:
            cursor = self.db.cursor(dictionary=True)
            
            query = """
                SELECT sender, message, intent, confidence, current_page, 
                       created_at as timestamp
                FROM chat_conversations 
                WHERE session_id = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            """
            
            cursor.execute(query, (session_id, limit))
            history = cursor.fetchall()
            cursor.close()
            
            # Convert timestamps to ISO format and reverse order (oldest first)
            for msg in history:
                if msg['timestamp']:
                    msg['timestamp'] = msg['timestamp'].isoformat()
            
            return list(reversed(history))
            
        except mysql.connector.Error as e:
            logger.error(f"❌ Failed to retrieve chat history: {e}")
            return []
    
    def get_user_recent_sessions(self, user_email: str, limit: int = 5) -> List[Dict]:
        """
        Get recent chat sessions for a specific user.
        
        Args:
            user_email: Customer email address
            limit: Number of recent sessions to retrieve
            
        Returns:
            List of session information
        """
        if not self.db or not user_email:
            return []
            
        try:
            cursor = self.db.cursor(dictionary=True)
            
            query = """
                SELECT s.session_id, s.first_message_at, s.last_message_at,
                       s.total_messages, s.primary_intent, s.resolution_status
                FROM chat_sessions s
                WHERE s.user_email = %s
                ORDER BY s.last_message_at DESC
                LIMIT %s
            """
            
            cursor.execute(query, (user_email, limit))
            sessions = cursor.fetchall()
            cursor.close()
            
            return sessions
            
        except mysql.connector.Error as e:
            logger.error(f"❌ Failed to retrieve user sessions: {e}")
            return []
    
    def _update_session_info(self, session_id: str, user_email: str = None, sender: str = 'user'):
        """
        Update session information in chat_sessions table.
        
        Args:
            session_id: Session identifier
            user_email: Customer email (optional)
            sender: Message sender type
        """
        try:
            cursor = self.db.cursor()
            
            # Check if session exists
            cursor.execute("SELECT id FROM chat_sessions WHERE session_id = %s", (session_id,))
            session_exists = cursor.fetchone()
            
            if not session_exists:
                # Create new session record
                session_type = 'admin' if 'admin' in sender else 'customer'
                query = """
                    INSERT INTO chat_sessions 
                    (session_id, user_email, session_type, first_message_at, 
                     last_message_at, total_messages)
                    VALUES (%s, %s, %s, NOW(), NOW(), 1)
                """
                cursor.execute(query, (session_id, user_email, session_type))
            else:
                # Update existing session
                query = """
                    UPDATE chat_sessions 
                    SET last_message_at = NOW(), 
                        total_messages = total_messages + 1,
                        user_email = COALESCE(%s, user_email)
                    WHERE session_id = %s
                """
                cursor.execute(query, (user_email, session_id))
            
            cursor.close()
            
        except mysql.connector.Error as e:
            logger.error(f"❌ Failed to update session info: {e}")
    
    def get_chat_analytics(self, days: int = 7) -> Dict:
        """
        Get chat analytics for the specified number of days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary containing analytics data
        """
        if not self.db:
            return {}
            
        try:
            cursor = self.db.cursor(dictionary=True)
            analytics = {}
            
            # Total messages and sessions
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(DISTINCT user_email) as unique_users,
                    AVG(response_time_ms) as avg_response_time
                FROM chat_conversations 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            """, (days,))
            analytics['summary'] = cursor.fetchone()
            
            # Popular intents
            cursor.execute("""
                SELECT intent, COUNT(*) as count
                FROM chat_conversations 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                AND intent IS NOT NULL
                GROUP BY intent
                ORDER BY count DESC
                LIMIT 10
            """, (days,))
            analytics['popular_intents'] = cursor.fetchall()
            
            # Daily message volume
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as message_count,
                    COUNT(DISTINCT session_id) as session_count
                FROM chat_conversations 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """, (days,))
            analytics['daily_volume'] = cursor.fetchall()
            
            cursor.close()
            return analytics
            
        except mysql.connector.Error as e:
            logger.error(f"❌ Failed to get chat analytics: {e}")
            return {}

# ================================================================
# CHATBOT ENGINE - MAIN AI PROCESSING CLASS
# ================================================================
class ChatbotEngine:
    """
    Core AI processing engine for the WebStore chatbot.
    
    This class manages:
    - Intent analysis and understanding
    - Conversation history and context
    - AI response generation
    - Integration with knowledge base and databases
    """
    
    def __init__(self, knowledge_base, mysql_db, gemini_model):
        """
        Initialize the chatbot engine with all required components.
        
        Args:
            knowledge_base: WebStoreKnowledgeBase instance
            mysql_db: MySQL database connection
            gemini_model: Google Gemini AI model instance
        """
        self.kb = knowledge_base          # Knowledge base for information retrieval
        self.db = mysql_db               # Database for real-time data
        self.ai_model = gemini_model     # AI model for natural language processing
        
        # Initialize database chat history manager
        self.chat_history = ChatHistoryManager(mysql_db)
        
        # Keep minimal in-memory cache for performance (last 5 messages per session)
        self.message_cache = {}
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve conversation history from database with caching.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        # Try cache first for recent messages
        if session_id in self.message_cache:
            cached_messages = self.message_cache[session_id]
            if len(cached_messages) >= min(limit, 5):
                return cached_messages[-limit:]
        
        # Fetch from database
        history = self.chat_history.get_session_history(session_id, limit)
        
        # Update cache with recent messages
        self.message_cache[session_id] = history[-5:] if history else []
        
        return history
    
    def save_message_to_history(self, session_id: str, sender: str, message: str,
                               intent: str = None, confidence: float = None,
                               user_email: str = None, current_page: str = None,
                               response_time_ms: int = None) -> bool:
        """
        Save a message to both database and cache.
        
        Args:
            session_id: Session identifier
            sender: Message sender ('user', 'bot', 'admin', 'admin_bot')
            message: Message content
            intent: Detected intent (optional)
            confidence: AI confidence score (optional)
            user_email: Customer email (optional)
            current_page: Current page (optional)
            response_time_ms: Response time (optional)
            
        Returns:
            Boolean indicating success
        """
        # Save to database
        success = self.chat_history.save_message(
            session_id, sender, message, intent, confidence,
            user_email, current_page, response_time_ms
        )
        
        if success:
            # Update in-memory cache
            message_obj = {
                'sender': sender,
                'message': message,
                'intent': intent,
                'confidence': confidence,
                'current_page': current_page,
                'timestamp': datetime.now().isoformat()
            }
            
            if session_id not in self.message_cache:
                self.message_cache[session_id] = []
            
            self.message_cache[session_id].append(message_obj)
            
            # Keep cache small (last 5 messages only)
            if len(self.message_cache[session_id]) > 5:
                self.message_cache[session_id] = self.message_cache[session_id][-5:]
        
        return success
    
    def analyze_intent(self, message: str, history: List[Dict] = None) -> Dict:
        """
        Analyze user's intent using advanced AI and pattern matching.
        
        This method:
        1. Uses regex patterns for quick detection of tracking numbers/order IDs
        2. Employs Google Gemini AI for complex intent analysis
        3. Considers conversation history for context
        4. Returns structured intent data with entities
        
        Args:
            message: User's input message
            history: Previous conversation messages for context
            
        Returns:
            Dict containing intent, entities, and confidence score
        """
        try:
            # First, try regex-based extraction for tracking numbers and order IDs
            tracking_number_match = re.search(r'TRK\d{6}', message.upper())
            order_id_match = re.search(r'#?(\d+)', message)
            
            context = ""
            if history:
                context = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in history[-3:]])
            
            prompt = f"""
            Analyze this e-commerce customer query and return ONLY a JSON response:

            {{
                "intent": "one of: product_search, order_status, account_info, website_info, support_request, general_chat",
                "entities": {{
                    "product_keywords": ["extracted product terms"],
                    "price_range": [min, max] or null,
                    "category": "product category" or null,
                    "email": "customer email" or null,
                    "order_id": "order ID without # symbol" or null,
                    "tracking_number": "tracking number (TRK followed by digits)" or null
                }},
                "confidence": 0.8
            }}

            Customer Query: "{message}"
            Conversation Context: {context}

            Examples:
            "Show me phones under $500" → {{"intent": "product_search", "entities": {{"product_keywords": ["phones"], "price_range": [0, 500], "category": null}}}}
            "Track my order #12345" → {{"intent": "order_status", "entities": {{"order_id": "12345"}}}}
            "Track TRK647975" → {{"intent": "order_status", "entities": {{"tracking_number": "TRK647975"}}}}
            "My tracking number is TRK123456" → {{"intent": "order_status", "entities": {{"tracking_number": "TRK123456"}}}}
            "How does checkout work?" → {{"intent": "website_info", "entities": {{}}}}
            """
            
            response = self.ai_model.generate_content(prompt)
            intent_data = json.loads(self.clean_ai_response(response.text))
            
            # Override with regex matches if found and not already detected
            if tracking_number_match and not intent_data.get('entities', {}).get('tracking_number'):
                intent_data['intent'] = 'order_status'
                intent_data['entities']['tracking_number'] = tracking_number_match.group(0)
            elif order_id_match and not intent_data.get('entities', {}).get('order_id') and 'track' in message.lower():
                intent_data['intent'] = 'order_status'
                intent_data['entities']['order_id'] = order_id_match.group(1)
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            
            # Enhanced fallback intent analysis using keyword matching
            message_lower = message.lower()
            fallback_intent = "general_chat"
            entities = {}
            
            # Basic keyword-based intent detection as fallback
            if any(word in message_lower for word in ['track', 'order', 'shipping', 'delivery']):
                fallback_intent = "order_status"
            elif any(word in message_lower for word in ['product', 'search', 'find', 'looking for']):
                fallback_intent = "product_search"
            elif any(word in message_lower for word in ['return', 'refund', 'exchange']):
                fallback_intent = "return_policy"
            elif any(word in message_lower for word in ['cart', 'checkout', 'payment']):
                fallback_intent = "shopping_cart"
            elif any(word in message_lower for word in ['contact', 'support', 'help']):
                fallback_intent = "customer_support"
            
            return {
                "intent": fallback_intent,
                "entities": entities,
                "confidence": 0.3  # Lower confidence for fallback
            }
    
    def clean_ai_response(self, text: str) -> str:
        """Clean AI response to extract JSON or clean text"""
        # Remove markdown code blocks
        text = re.sub(r'^```(json)?\n|```$', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # Remove quotes if they wrap entire response
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
            
        return text

# ================================================================
# 13. INITIALIZE CORE COMPONENTS
# ================================================================
# Create instances of the main system components

# Initialize the comprehensive knowledge base
kb = WebStoreKnowledgeBase(mysql_db, knowledge_collection, embedding_model)

# Initialize the main chatbot processing engine
chatbot = ChatbotEngine(kb, mysql_db, gemini_model)

# ================================================================
# 14. API ENDPOINTS - CHATBOT INITIALIZATION
# ================================================================

@app.route('/api/chat/init', methods=['POST'])
def initialize_chatbot():
    """
    Initialize the chatbot knowledge base with fresh data.
    
    This endpoint:
    1. Rebuilds the vector embeddings from current database content
    2. Updates the AI knowledge base with latest products and policies
    3. Prepares the system for optimal customer service
    
    Returns:
        JSON response indicating success or failure of initialization
    """
    try:
        logger.info("🔄 Starting chatbot initialization...")
        kb.build_knowledge_embeddings()
        logger.info("✅ Chatbot initialization completed successfully")
        
        return jsonify({
            "status": "success",
            "message": "Chatbot knowledge base initialized successfully"
        })
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return jsonify({
            "status": "error", 
            "message": "Failed to initialize chatbot"
        }), 500

# ================================================================
# 15. API ENDPOINTS - MAIN CHAT INTERFACE
# ================================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint for customer interactions.
    
    This endpoint handles:
    1. Message processing and intent analysis
    2. Context extraction from current page/product
    3. Knowledge base search and database queries
    4. AI response generation with full context
    5. Conversation history management
    
    Expected JSON payload:
    {
        "message": "user's message",
        "session_id": "unique session identifier",
        "user_email": "customer email (optional)",
        "current_page": "current website page",
        "product_context": {"product details if on product page"}
    }
    
    Returns:
        JSON response with AI-generated answer and session information
    """
    try:
        # ================================================================
        # REQUEST PROCESSING AND VALIDATION
        # ================================================================
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id') or str(uuid.uuid4())
        user_email = data.get('user_email')  # Optional user identification
        current_page = data.get('current_page', '')
        product_context = data.get('product_context', {})  # Single product page context
        
        # Enhanced product context extraction
        # If product ID is provided but details missing, fetch from database
        if product_context.get('id') and not product_context.get('name'):
            detailed_product = get_product_details(product_context['id'])
            if detailed_product:
                product_context.update({
                    'name': detailed_product.get('product_name'),
                    'type': detailed_product.get('product_type'),
                    'price': detailed_product.get('product_price'),
                    'description': detailed_product.get('product_discription'),
                    'stock_quantity': detailed_product.get('stock_quantity'),
                    'sizes': ', '.join(detailed_product.get('sizes', [])),
                    'image': detailed_product.get('product_image')
                })
        
        # Validate that user provided a message
        if not user_message:
            return jsonify({
                "response": "Please enter a message.",
                "session_id": session_id
            })
        
        # ================================================================
        # CONVERSATION HISTORY MANAGEMENT (DATABASE-BASED)
        # ================================================================
        # Get conversation history from database
        history = chatbot.get_conversation_history(session_id, limit=10)
        
        # Record start time for response time tracking
        start_time = datetime.now()
        
        # Save user message to database
        chatbot.save_message_to_history(
            session_id=session_id,
            sender="user", 
            message=user_message,
            user_email=user_email,
            current_page=current_page
        )
        
        # ================================================================
        # AI PROCESSING PIPELINE
        # ================================================================
        
        # Step 1: Analyze user intent using AI and pattern matching
        intent_data = chatbot.analyze_intent(user_message, history)
        
        # Step 2: Check for navigation intent (product/page navigation)
        navigation_intent = analyze_navigation_intent(user_message)
        if navigation_intent:
            intent_data.update(navigation_intent)
        
        # Search knowledge base
        knowledge_results = kb.search_knowledge(user_message, n_results=3)
        
        # Generate response with product context
        response = generate_response(
            user_message=user_message,
            intent=intent_data,
            knowledge=knowledge_results,
            user_email=user_email,
            history=history,
            current_page=current_page,
            product_context=product_context
        )
        
        # ================================================================
        # RESPONSE FINALIZATION AND STORAGE
        # ================================================================
        
        # Calculate response time
        end_time = datetime.now()
        response_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Save bot response to database
        chatbot.save_message_to_history(
            session_id=session_id,
            sender="bot", 
            message=response,
            intent=intent_data.get('intent'),
            confidence=intent_data.get('confidence'),
            user_email=user_email,
            current_page=current_page,
            response_time_ms=response_time_ms
        )
        
        # Return structured response
        return jsonify({
            "response": response,
            "session_id": session_id,
            "intent": intent_data.get('intent'),
            "confidence": intent_data.get('confidence', 0),
            "response_time_ms": response_time_ms
        })
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        return jsonify({
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team.",
            "session_id": session_id or str(uuid.uuid4())
        }), 500

def generate_response(user_message: str, intent: Dict, knowledge: List, user_email: str = None, history: List = None, current_page: str = "", product_context: Dict = None) -> str:
    """Generate AI response using all available context"""
    try:
        # Build context from knowledge base
        context_parts = []
        
        # Add relevant knowledge
        for k in knowledge:
            context_parts.append(f"Knowledge: {k['content']}")
        
        # Add single product context if available
        if product_context and current_page.lower() == 'single product page':
            context_parts.append("**CURRENT PRODUCT CONTEXT:**")
            if product_context.get('name'):
                context_parts.append(f"Product Name: {product_context['name']}")
            if product_context.get('type'):
                context_parts.append(f"Category: {product_context['type']}")
            if product_context.get('price'):
                context_parts.append(f"Price: ${product_context['price']}")
            if product_context.get('sizes'):
                context_parts.append(f"Available Sizes: {product_context['sizes']}")
            if product_context.get('description'):
                context_parts.append(f"Description: {product_context['description']}")
            if product_context.get('stock_quantity'):
                stock_status = "In Stock" if int(product_context['stock_quantity']) > 0 else "Out of Stock"
                context_parts.append(f"Stock Status: {stock_status} ({product_context['stock_quantity']} units)")
            if product_context.get('image'):
                context_parts.append(f"Product Image: {product_context['image']}")
        
        # Add database context based on intent
        if intent['intent'] == 'product_search':
            products = search_products(intent.get('entities', {}))
            if products:
                context_parts.append("Available Products:")
                for product in products[:5]:  # Limit to 5 products
                    context_parts.append(f"- {product.get('product_name')} (${product.get('product_price')}) - {product.get('product_discription')}")
        
        elif intent['intent'] == 'order_status':
            order_id = intent.get('entities', {}).get('order_id')
            tracking_number = intent.get('entities', {}).get('tracking_number')
            order_details = None
            
            # Try to search by tracking number first
            if tracking_number:
                order_details = get_order_by_tracking_number(tracking_number)
                if order_details:
                    context_parts.append("**ORDER TRACKING INFORMATION (by Tracking Number):**")
                else:
                    context_parts.append(f"Tracking number '{tracking_number}' not found in our system. Please check the tracking number.")
            
            # If no tracking number or tracking search failed, try order ID
            elif order_id:
                order_details = get_order_by_id(order_id)
                if order_details:
                    context_parts.append("**ORDER TRACKING INFORMATION (by Order ID):**")
                else:
                    context_parts.append(f"Order #{order_id} not found in our system. Please check the order number.")
            
            # If we found order details, add them to context
            if order_details:
                context_parts.append(f"Order #{order_details.get('order_id')}")
                context_parts.append(f"Customer: {order_details.get('first_name', '')} {order_details.get('last_name', '')}")
                context_parts.append(f"Order Date: {order_details.get('order_date')}")
                context_parts.append(f"Total Amount: ${order_details.get('total_amount', 0):.2f}")
                context_parts.append(f"Status: {order_details.get('tracking_status', 'Processing')}")
                context_parts.append(f"Tracking: {order_details.get('tracking_message', 'Order is being processed')}")
                
                if order_details.get('tracking_number'):
                    context_parts.append(f"Tracking Number: {order_details.get('tracking_number')}")
                
                if order_details.get('estimated_delivery'):
                    context_parts.append(f"Estimated Delivery: {order_details.get('estimated_delivery')}")
                
                if order_details.get('items'):
                    context_parts.append("Ordered Items:")
                    for item in order_details['items']:
                        context_parts.append(f"- {item.get('product_name')} x{item.get('quantity')} - ${item.get('total_price', 0):.2f}")
                
                if order_details.get('payment'):
                    payment = order_details['payment']
                    context_parts.append(f"Payment: {payment.get('payment_method')} - {payment.get('payment_status', 'Completed')}")
                
                # Add shipping address if available
                if order_details.get('shipping_address'):
                    context_parts.append(f"Shipping Address: {order_details.get('shipping_address')}, {order_details.get('city')}, {order_details.get('state')} {order_details.get('postal_code')}")
            
            elif user_email:
                # Search by customer email if no order ID or tracking number provided
                orders = get_customer_orders(user_email)
                if orders:
                    context_parts.append("Customer Orders:")
                    for order in orders[-3:]:  # Last 3 orders
                        context_parts.append(f"- Order #{order.get('order_id')}: ${order.get('total_amount')} on {order.get('order_date')}")
            else:
                context_parts.append("To track your order, please provide:")
                context_parts.append("• Your order number (e.g., #12345)")
                context_parts.append("• Your tracking number (e.g., TRK647975)")
                context_parts.append("• Or your email address")
        
        elif intent.get('intent') == 'navigation':
            # Handle navigation requests
            if intent.get('type') == 'product_navigation':
                products = intent.get('products', [])
                context_parts.append("**PRODUCT NAVIGATION:**")
                context_parts.append(f"Found products matching '{intent.get('search_term')}':")
                for product in products:
                    product_url = f"/sproductpage.html?id={product['product_id']}"
                    context_parts.append(f"- {product['product_name']} (${product['product_price']}) - Link: {product_url}")
                    
            elif intent.get('type') == 'page_navigation':
                context_parts.append("**PAGE NAVIGATION:**")
                context_parts.append(f"Page: {intent.get('page_name')} - URL: {intent.get('page_url')}")
        
        # Build conversation context
        recent_history = ""
        if history and len(history) > 1:
            recent_history = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in history[-3:-1]])
        
        # Generate AI response
        context_text = "\n".join(context_parts)
        
        # Enhanced prompt for product-specific context
        if product_context and current_page.lower() == 'single product page':
            prompt = f"""
            You are a helpful product specialist for WebStore, currently helping a customer on a specific product page.
            
            CURRENT CONTEXT: Single Product Page
            Product Details: {product_context}
            
            Customer Query: "{user_message}"
            Intent: {intent.get('intent')}
            
            Available Context:
            {context_text}
            
            Recent Conversation:
            {recent_history}
            
            Guidelines for Product Page:
            - Focus on the specific product the customer is viewing
            - Provide detailed information about sizes, pricing, stock availability
            - Help with product-related questions (sizing, fit, care instructions, etc.)
            - Suggest complementary products when appropriate
            - Guide customers through add-to-cart process
            - Mention login requirement for adding to cart
            - Be enthusiastic about the product while being honest about its features
            - If asked about other products, relate back to the current product
            - Provide accurate stock and availability information
            
            Response should be conversational, helpful, and product-focused.
            """
        elif intent.get('intent') == 'navigation':
            # Special prompt for navigation requests
            prompt = f"""
            You are a helpful navigation assistant for WebStore. A customer is asking to navigate to a specific product or page.
            
            Customer Request: "{user_message}"
            Navigation Context: {context_text}
            
            Guidelines for Navigation:
            - If products were found, provide clickable links to the product pages
            - Format product links as: "Click here to view [Product Name]: /sproductpage.html?id=[product_id]"
            - If it's a page navigation, provide the direct page link
            - Be enthusiastic about helping them find what they're looking for
            - If multiple products match, present the top options clearly
            - Mention that they can click the links to navigate directly
            - If no exact matches found, suggest similar or alternative products
            
            Make the response actionable with clear navigation instructions.
            """
        else:
            prompt = f"""
            You are a helpful customer service AI for WebStore, an e-commerce clothing store. 
            Provide accurate, friendly, and helpful responses based on the available context.

            User Query: "{user_message}"
            Intent: {intent.get('intent')}
            Current Page: {current_page}
            
            Available Context:
            {context_text}
            
            Recent Conversation:
            {recent_history}
            
            Guidelines:
            - Be friendly, professional, and concise
            - Use the provided context to give accurate information
            - If you don't have specific information, direct them to contact support
            - For product queries, mention specific products from the database
            - For order queries, use actual order information if available
            - Don't make up information not in the context
            """
        
        response = gemini_model.generate_content(prompt)
        return chatbot.clean_ai_response(response.text)
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        
        # Provide specific error responses based on error type
        error_str = str(e).lower()
        if "404" in error_str and "model" in error_str:
            return "I'm currently having an issue with my AI model. Our technical team has been notified. In the meantime, please contact our support team at support@webstore.com for immediate assistance."
        elif "quota" in error_str or "limit" in error_str:
            return "I'm currently experiencing high demand. Please try again in a few moments, or contact our support team for immediate assistance."
        elif "unauthorized" in error_str or "api key" in error_str:
            return "I'm having authentication issues with my AI service. Please contact our support team for assistance while we resolve this."
        else:
            return "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question or contact our support team for assistance."

def generate_admin_response(user_message: str, intent: Dict, knowledge: List, admin_context: Dict, history: List = None, current_page: str = "Admin Panel") -> str:
    """Generate AI response for admin users with business analytics context"""
    try:
        # Build context from knowledge base
        context_parts = []
        
        # Add relevant knowledge
        for k in knowledge:
            context_parts.append(f"Knowledge: {k['content']}")
        
        # Add comprehensive admin analytics context
        if isinstance(admin_context, dict) and 'error' not in admin_context:
            # Today's performance
            today_data = admin_context.get('today', {})
            yesterday_data = admin_context.get('yesterday', {})
            this_month_data = admin_context.get('this_month', {})
            
            if today_data:
                context_parts.append(f"**TODAY'S PERFORMANCE:**")
                context_parts.append(f"- Orders: {today_data.get('orders_today', 0)}")
                context_parts.append(f"- Revenue: ${today_data.get('revenue_today', 0):.2f}")
                context_parts.append(f"- Average Order Value: ${today_data.get('avg_order_today', 0):.2f}")
                
                if yesterday_data:
                    orders_change = today_data.get('orders_today', 0) - yesterday_data.get('orders_yesterday', 0)
                    revenue_change = today_data.get('revenue_today', 0) - yesterday_data.get('revenue_yesterday', 0)
                    context_parts.append(f"- Change from Yesterday: {orders_change:+d} orders, ${revenue_change:+.2f} revenue")
            
            if this_month_data:
                context_parts.append(f"**THIS MONTH'S PERFORMANCE:**")
                context_parts.append(f"- Total Orders: {this_month_data.get('orders_this_month', 0)}")
                context_parts.append(f"- Total Revenue: ${this_month_data.get('revenue_this_month', 0):.2f}")
                context_parts.append(f"- Average Order Value: ${this_month_data.get('avg_order_this_month', 0):.2f}")
            
            # Top products
            top_products = admin_context.get('top_products', [])
            if top_products:
                context_parts.append("**TOP SELLING PRODUCTS (Last 30 Days):**")
                for i, product in enumerate(top_products[:5], 1):
                    context_parts.append(f"{i}. {product.get('product_name')} - {product.get('units_sold')} units sold - ${product.get('product_revenue', 0):.2f} revenue - Category: {product.get('product_type', 'N/A')}")
            
            # Critical stock alerts
            low_stock = admin_context.get('low_stock', [])
            if low_stock:
                critical_items = [item for item in low_stock if item.get('stock_quantity', 0) <= 5]
                out_of_stock = [item for item in low_stock if item.get('stock_quantity', 0) == 0]
                
                if out_of_stock:
                    context_parts.append("**🚨 OUT OF STOCK ITEMS:**")
                    for item in out_of_stock[:5]:
                        context_parts.append(f"- {item.get('product_name')} - {item.get('product_type', 'N/A')} - ${item.get('product_price', 0):.2f}")
                
                if critical_items:
                    context_parts.append("**⚠️ CRITICAL STOCK ALERTS (≤5 units):**")
                    for item in critical_items[:5]:
                        context_parts.append(f"- {item.get('product_name')} - {item.get('stock_quantity')} remaining - {item.get('product_type', 'N/A')}")
            
            # Customer metrics
            customer_data = admin_context.get('customer_metrics', {})
            total_customers = admin_context.get('total_customers', {})
            if customer_data:
                context_parts.append("**CUSTOMER ANALYTICS:**")
                context_parts.append(f"- Active Customers (30 days): {customer_data.get('active_customers', 0)}")
                context_parts.append(f"- Total Registered Customers: {total_customers.get('total_customers', 0)}")
                context_parts.append(f"- Average Customer Order Value: ${customer_data.get('avg_customer_value', 0):.2f}")
                context_parts.append(f"- Customers Who Ordered Today: {customer_data.get('customers_today', 0)}")
            
            # Category performance
            categories = admin_context.get('category_performance', [])
            if categories:
                context_parts.append("**PRODUCT CATEGORY PERFORMANCE:**")
                for cat in categories[:5]:
                    if cat.get('category_revenue', 0) > 0:
                        context_parts.append(f"- {cat.get('category', 'Unknown')}: {cat.get('units_sold', 0)} units, ${cat.get('category_revenue', 0):.2f} revenue")
            
            # Recent inquiries
            inquiries = admin_context.get('recent_inquiries', [])
            if inquiries:
                context_parts.append("**RECENT CUSTOMER INQUIRIES:**")
                for inquiry in inquiries[:3]:
                    context_parts.append(f"- {inquiry.get('subject', 'No Subject')} from {inquiry.get('name', 'Unknown')} ({inquiry.get('time_ago', 'Unknown time')})")
        
        # Build conversation context
        recent_history = ""
        if history and len(history) > 1:
            recent_history = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in history[-3:-1]])
        
        # Generate AI response with admin context
        context_text = "\n".join(context_parts)
        
        # Detect specific admin queries
        user_lower = user_message.lower()
        is_report_request = any(word in user_lower for word in ['report', 'analytics', 'performance', 'sales', 'today', 'revenue', 'summary'])
        is_stock_request = any(word in user_lower for word in ['stock', 'inventory', 'low stock', 'out of stock', 'products'])
        is_customer_request = any(word in user_lower for word in ['customer', 'clients', 'users'])
        
        # Enhanced admin query handling
        enhanced_context = ""
        if is_stock_request:
            enhanced_context = "\n\n**INVENTORY ANALYSIS REQUEST DETECTED**\n"
            enhanced_context += "User is asking about inventory/stock status. Focus on:\n"
            enhanced_context += "- Current stock levels for all products\n"
            enhanced_context += "- Critical low stock alerts (≤5 units)\n"
            enhanced_context += "- Out of stock items requiring restocking\n"
            enhanced_context += "- Product categories with stock issues\n"
            enhanced_context += "- Specific product quantities and recommendations\n"
        
        prompt = f"""
        You are an advanced AI business consultant for WebStore's admin panel on the {current_page}.
        Provide strategic, data-driven insights with real numbers and actionable recommendations.

        Admin Query: "{user_message}"
        Intent: {intent.get('intent')}
        Current Page: {current_page}
        {enhanced_context}
        
        Real-Time Business Data:
        {context_text}
        
        Recent Conversation:
        {recent_history}
        
        Admin Guidelines:
        - Use REAL NUMBERS from the business data provided above
        - For "today's report" or similar: provide comprehensive daily summary with actual figures
        - For stock/inventory inquiries: provide detailed inventory analysis with specific product names, quantities, and stock status
        - For customer questions: provide actual customer counts and behavior data  
        - Format with clear sections using **bold headings**
        - Use bullet points for easy scanning
        - Highlight critical alerts with emojis (🚨 ⚠️ 📈 📉 💰)
        - Be specific and actionable - mention exact products, numbers, and recommendations
        - If data shows concerning trends, highlight them prominently
        - Always conclude with 2-3 specific action items
        - When discussing inventory, include product names, current stock levels, and restocking recommendations
        """
        
        response = gemini_model.generate_content(prompt)
        return chatbot.clean_ai_response(response.text)
        
    except Exception as e:
        logger.error(f"Admin response generation failed: {e}")
        return "I'm experiencing technical difficulties accessing the admin analytics. Please try again or check the system status."

def search_products(entities: Dict) -> List[Dict]:
    """Search products in MySQL database"""
    if not mysql_db:
        return []
    
    try:
        cursor = mysql_db.cursor(dictionary=True)
        
        query = """
        SELECT p.*, i.stock_quantity 
        FROM product p 
        LEFT JOIN inventory i ON p.product_id = i.product_id
        WHERE 1=1
        """
        params = []
        
        # Add keyword search
        keywords = entities.get('product_keywords', [])
        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append("(p.product_name LIKE %s OR p.product_discription LIKE %s)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
            
            if keyword_conditions:
                query += " AND (" + " OR ".join(keyword_conditions) + ")"
        
        # Add category filter
        if entities.get('category'):
            query += " AND p.product_type = %s"
            params.append(entities['category'])
        
        # Add price range filter
        price_range = entities.get('price_range')
        if price_range and len(price_range) == 2:
            query += " AND p.product_price BETWEEN %s AND %s"
            params.extend(price_range)
        
        query += " ORDER BY p.product_ranking DESC LIMIT 10"
        
        cursor.execute(query, params)
        products = cursor.fetchall()
        cursor.close()
        
        return products
        
    except mysql.connector.Error as e:
        logger.error(f"Product search failed: {e}")
        return []

def get_product_details(product_id):
    """Get detailed product information including sizes and stock"""
    if not mysql_db:
        return None
        
    try:
        cursor = mysql_db.cursor(dictionary=True)
        
        # Get main product details
        product_query = """
        SELECT p.*, i.stock_quantity 
        FROM product p 
        LEFT JOIN inventory i ON p.product_id = i.product_id 
        WHERE p.product_id = %s
        """
        cursor.execute(product_query, (product_id,))
        product = cursor.fetchone()
        
        if not product:
            cursor.close()
            return None
            
        # Get available sizes
        sizes_query = """
        SELECT size_name FROM product_sizes WHERE product_id = %s
        """
        cursor.execute(sizes_query, (product_id,))
        sizes = cursor.fetchall()
        
        product['sizes'] = [size['size_name'] for size in sizes] if sizes else []
        cursor.close()
        
        return product
        
    except mysql.connector.Error as e:
        logger.error(f"Error getting product details: {e}")
        return None

def get_customer_orders(email: str) -> List[Dict]:
    """Get customer orders from database"""
    if not mysql_db:
        return []
    
    try:
        cursor = mysql_db.cursor(dictionary=True)
        
        query = """
        SELECT o.*, c.first_name, c.last_name
        FROM `order` o
        JOIN customer c ON o.customer_id = c.customer_id
        WHERE c.email = %s
        ORDER BY o.order_date DESC
        LIMIT 10
        """
        
        cursor.execute(query, (email,))
        orders = cursor.fetchall()
        cursor.close()
        
        return orders
        
    except mysql.connector.Error as e:
        logger.error(f"Order search failed: {e}")
        return []

def get_order_by_id(order_id: str) -> Optional[Dict]:
    """Get specific order details by order ID with proper tracking logic"""
    if not mysql_db:
        return None
    
    try:
        cursor = mysql_db.cursor(dictionary=True)
        
        # Remove # symbol if present and convert to int
        clean_order_id = order_id.replace('#', '').strip()
        
        # Get order details with customer info and shipping details
        order_query = """
        SELECT o.*, c.first_name, c.last_name, c.email, c.phone,
               s.tracking_number, s.shipping_address, s.city, s.state, s.country, 
               s.postal_code, s.shipping_date
        FROM `order` o
        JOIN customer c ON o.customer_id = c.customer_id
        LEFT JOIN shipping s ON o.order_id = s.order_id
        WHERE o.order_id = %s
        """
        
        cursor.execute(order_query, (clean_order_id,))
        order = cursor.fetchone()
        
        if not order:
            cursor.close()
            return None
        
        # Get order items
        items_query = """
        SELECT oi.*, p.product_name, p.product_image
        FROM order_items oi
        JOIN product p ON oi.product_id = p.product_id
        WHERE oi.order_id = %s
        """
        
        cursor.execute(items_query, (clean_order_id,))
        items = cursor.fetchall()
        
        # Get payment info
        payment_query = """
        SELECT payment_method, payment_amount, payment_status, payment_date
        FROM payment
        WHERE order_id = %s
        """
        
        cursor.execute(payment_query, (clean_order_id,))
        payment = cursor.fetchone()
        
        cursor.close()
        
        # Add items and payment to order
        order['items'] = items or []
        order['payment'] = payment
        
        # Enhanced tracking status logic
        from datetime import datetime, timedelta
        order_date = order.get('order_date')
        shipping_date = order.get('shipping_date')
        tracking_number = order.get('tracking_number')
        
        if order_date:
            days_since_order = (datetime.now() - order_date).days
            
            # Use actual tracking number if available
            if tracking_number:
                order['tracking_number'] = tracking_number
                order['has_tracking'] = True
            else:
                # Generate tracking number format: TRK + order_id + random suffix
                order['tracking_number'] = f"TRK{clean_order_id}{order_date.strftime('%m%d')}"
                order['has_tracking'] = False
            
            # Determine status based on shipping date and order age
            if shipping_date:
                days_since_shipping = (datetime.now() - shipping_date).days
                if days_since_shipping >= 5:
                    order['tracking_status'] = 'Delivered'
                    order['tracking_message'] = f'Your order was delivered on {(shipping_date + timedelta(days=5)).strftime("%B %d, %Y")}.'
                elif days_since_shipping >= 1:
                    estimated_delivery = shipping_date + timedelta(days=5)
                    order['tracking_status'] = 'In Transit'
                    order['tracking_message'] = f'Your order is on its way! Expected delivery: {estimated_delivery.strftime("%B %d, %Y")}'
                    order['estimated_delivery'] = estimated_delivery.strftime('%Y-%m-%d')
                else:
                    order['tracking_status'] = 'Shipped'
                    order['tracking_message'] = f'Your order shipped on {shipping_date.strftime("%B %d, %Y")} and is being processed for delivery.'
            else:
                # No shipping date - determine by order age
                if days_since_order == 0:
                    order['tracking_status'] = 'Order Confirmed'
                    order['tracking_message'] = 'Your order has been confirmed and payment processed. We\'re preparing your items.'
                elif days_since_order == 1:
                    order['tracking_status'] = 'Processing'
                    order['tracking_message'] = 'Your order is being processed and will be shipped within 24 hours.'
                elif days_since_order <= 3:
                    order['tracking_status'] = 'Ready to Ship'
                    order['tracking_message'] = 'Your order is packed and ready to ship. You\'ll receive tracking info soon.'
                    estimated_shipping = order_date + timedelta(days=3)
                    order['estimated_shipping'] = estimated_shipping.strftime('%Y-%m-%d')
                else:
                    # Older orders should be shipped by now
                    order['tracking_status'] = 'Shipped'
                    order['tracking_message'] = 'Your order has been shipped. Delivery expected within 3-5 business days.'
                    estimated_delivery = order_date + timedelta(days=7)
                    order['estimated_delivery'] = estimated_delivery.strftime('%Y-%m-%d')
        
        return order
        
    except mysql.connector.Error as e:
        logger.error(f"Order tracking failed: {e}")
        return None

def get_order_by_tracking_number(tracking_number: str) -> Optional[Dict]:
    """Get order details by tracking number"""
    if not mysql_db:
        return None
    
    try:
        cursor = mysql_db.cursor(dictionary=True)
        
        # Clean tracking number (remove any extra spaces)
        clean_tracking = tracking_number.strip()
        
        # Get order details using tracking number
        order_query = """
        SELECT o.*, c.first_name, c.last_name, c.email, c.phone,
               s.tracking_number, s.shipping_address, s.city, s.state, s.country, 
               s.postal_code, s.shipping_date
        FROM `order` o
        JOIN customer c ON o.customer_id = c.customer_id
        JOIN shipping s ON o.order_id = s.order_id
        WHERE s.tracking_number = %s
        """
        
        cursor.execute(order_query, (clean_tracking,))
        order = cursor.fetchone()
        
        if not order:
            cursor.close()
            return None
        
        # Get order items
        items_query = """
        SELECT oi.*, p.product_name, p.product_image
        FROM order_items oi
        JOIN product p ON oi.product_id = p.product_id
        WHERE oi.order_id = %s
        """
        
        cursor.execute(items_query, (order['order_id'],))
        items = cursor.fetchall()
        
        # Get payment info
        payment_query = """
        SELECT payment_method, payment_amount, payment_status, payment_date
        FROM payment
        WHERE order_id = %s
        """
        
        cursor.execute(payment_query, (order['order_id'],))
        payment = cursor.fetchone()
        
        cursor.close()
        
        # Add items and payment to order
        order['items'] = items or []
        order['payment'] = payment
        
        # Enhanced tracking status logic using the same logic as get_order_by_id
        from datetime import datetime, timedelta
        order_date = order.get('order_date')
        shipping_date = order.get('shipping_date')
        tracking_number = order.get('tracking_number')
        
        if order_date:
            days_since_order = (datetime.now() - order_date).days
            
            # Use actual tracking number
            order['tracking_number'] = tracking_number
            order['has_tracking'] = True
            
            # Determine status based on shipping date and order age
            if shipping_date:
                days_since_shipping = (datetime.now() - shipping_date).days
                if days_since_shipping >= 5:
                    order['tracking_status'] = 'Delivered'
                    order['tracking_message'] = f'Your order was delivered on {(shipping_date + timedelta(days=5)).strftime("%B %d, %Y")}.'
                elif days_since_shipping >= 1:
                    estimated_delivery = shipping_date + timedelta(days=5)
                    order['tracking_status'] = 'In Transit'
                    order['tracking_message'] = f'Your order is on its way! Expected delivery: {estimated_delivery.strftime("%B %d, %Y")}'
                    order['estimated_delivery'] = estimated_delivery.strftime('%Y-%m-%d')
                else:
                    order['tracking_status'] = 'Shipped'
                    order['tracking_message'] = f'Your order shipped on {shipping_date.strftime("%B %d, %Y")} and is being processed for delivery.'
            else:
                # No shipping date - determine by order age
                if days_since_order == 0:
                    order['tracking_status'] = 'Order Confirmed'
                    order['tracking_message'] = 'Your order has been confirmed and payment processed. We\'re preparing your items.'
                elif days_since_order == 1:
                    order['tracking_status'] = 'Processing'
                    order['tracking_message'] = 'Your order is being processed and will be shipped within 24 hours.'
                elif days_since_order <= 3:
                    order['tracking_status'] = 'Ready to Ship'
                    order['tracking_message'] = 'Your order is packed and ready to ship. You\'ll receive tracking info soon.'
                    estimated_shipping = order_date + timedelta(days=3)
                    order['estimated_shipping'] = estimated_shipping.strftime('%Y-%m-%d')
                else:
                    # Older orders should be shipped by now
                    order['tracking_status'] = 'Shipped'
                    order['tracking_message'] = 'Your order has been shipped. Delivery expected within 3-5 business days.'
                    estimated_delivery = order_date + timedelta(days=7)
                    order['estimated_delivery'] = estimated_delivery.strftime('%Y-%m-%d')
        
        return order
        
    except mysql.connector.Error as e:
        logger.error(f"Tracking number search failed: {e}")
        return None

def analyze_navigation_intent(user_message: str) -> Optional[Dict]:
    """Analyze if user wants to navigate to a specific product or page"""
    # Product navigation keywords
    nav_patterns = [
        r"(?:move|go|navigate|take me|show|find|search)\s+(?:to\s+)?(.+?)(?:\s+page)?$",
        r"(?:i want to see|show me|find|search for)\s+(.+)",
        r"(?:where is|how to find)\s+(.+)",
        r"(?:open|visit)\s+(.+?)(?:\s+page)?$"
    ]
    
    user_message_lower = user_message.lower().strip()
    
    for pattern in nav_patterns:
        match = re.search(pattern, user_message_lower, re.IGNORECASE)
        if match:
            search_term = match.group(1).strip()
            
            # Remove common words
            search_term = re.sub(r'\b(the|a|an|page|product|item)\b', '', search_term).strip()
            
            if search_term:
                # Search for products matching the term
                products = search_products_by_name(search_term)
                
                if products:
                    return {
                        'type': 'product_navigation',
                        'search_term': search_term,
                        'products': products[:3],  # Return top 3 matches
                        'intent': 'navigation'
                    }
                else:
                    # Check for common page names
                    page_mappings = {
                        'shop': '/shop.html',
                        'cart': '/cart.html',
                        'checkout': '/checkout.html',
                        'about': '/about.html',
                        'contact': '/contact.html',
                        'blog': '/blog.html',
                        'wishlist': '/wishlist.html',
                        'account': '/user.html',
                        'profile': '/user.html',
                        'home': '/index.html',
                        'main': '/index.html'
                    }
                    
                    for page_key, page_url in page_mappings.items():
                        if page_key in search_term:
                            return {
                                'type': 'page_navigation',
                                'page_name': page_key,
                                'page_url': page_url,
                                'intent': 'navigation'
                            }
    
    return None

def search_products_by_name(search_term: str) -> List[Dict]:
    """Search products by name with fuzzy matching"""
    if not mysql_db:
        return []
    
    try:
        cursor = mysql_db.cursor(dictionary=True)
        
        # Search with LIKE pattern for partial matches
        search_query = """
        SELECT product_id, product_name, product_image, product_price, product_description
        FROM product
        WHERE product_name LIKE %s OR product_description LIKE %s
        ORDER BY 
            CASE 
                WHEN product_name LIKE %s THEN 1
                WHEN product_name LIKE %s THEN 2
                ELSE 3
            END,
            product_name
        LIMIT 10
        """
        
        exact_match = f"%{search_term}%"
        starts_with = f"{search_term}%"
        
        cursor.execute(search_query, (exact_match, exact_match, starts_with, exact_match))
        products = cursor.fetchall()
        cursor.close()
        
        return products
        
    except mysql.connector.Error as e:
        logger.error(f"Product search failed: {e}")
        return []

# ================================================================
# ADMIN PANEL CHATBOT INTEGRATION
# ================================================================

@app.route('/api/admin/chat', methods=['POST'])
def admin_chat():
    """
    Specialized chatbot endpoint for admin panel with business analytics.
    
    This endpoint provides:
    1. Real-time business analytics and reporting
    2. Sales performance analysis and trends
    3. Customer behavior insights
    4. Inventory management guidance
    5. Strategic business recommendations
    
    Returns comprehensive business intelligence responses for admin users.
    """
    try:
        # Extract admin request data
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'admin_session')
        current_page = data.get('current_page', 'Admin Panel')
        current_page = data.get('current_page', 'Admin Panel')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get admin analytics data for context
        admin_context = get_admin_analytics_data()
        
        # Admin-specific context and capabilities
        admin_system_prompt = f"""
        You are an advanced AI assistant for WebStore's admin panel. You are currently helping on the {current_page}.
        
        ADMIN CAPABILITIES:
        - Real-time business analytics and reporting
        - Sales performance analysis
        - Customer behavior insights
        - Inventory management guidance
        - Marketing strategy recommendations
        - Financial reporting and trends
        
        CURRENT BUSINESS DATA:
        {admin_context}
        
        ADMIN ROLE CONTEXT:
        - You have access to sensitive business data
        - Provide strategic insights and recommendations
        - Generate reports with actual numbers and trends
        - Help with business decision making
        - Offer data-driven advice for growth
        
        PAGE CONTEXT: {current_page}
        
        Respond as a professional business consultant with access to real-time data.
        Use actual numbers from the business data provided.
        Format responses with clear headings and bullet points for executive summaries.
        """
        
        # ================================================================
        # ADMIN CONVERSATION HISTORY MANAGEMENT (DATABASE-BASED)
        # ================================================================
        # Get admin conversation history from database
        history = chatbot.get_conversation_history(session_id, limit=10)
        
        # Record start time for response time tracking
        start_time = datetime.now()
        
        # Save admin message to database
        chatbot.save_message_to_history(
            session_id=session_id,
            sender="admin", 
            message=user_message,
            current_page=current_page
        )
        
        # Analyze intent with admin context
        intent_data = chatbot.analyze_intent(user_message, history)
        
        # Search knowledge base for admin-relevant information
        knowledge_results = kb.search_knowledge(user_message, n_results=3)
        
        # Generate admin-specific response with business analytics
        response = generate_admin_response(
            user_message=user_message,
            intent=intent_data,
            knowledge=knowledge_results,
            admin_context=admin_context,
            history=history,
            current_page=current_page
        )
        
        # ================================================================
        # ADMIN RESPONSE FINALIZATION AND STORAGE
        # ================================================================
        
        # Calculate response time
        end_time = datetime.now()
        response_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Save admin bot response to database
        chatbot.save_message_to_history(
            session_id=session_id,
            sender="admin_bot", 
            message=response,
            intent=intent_data.get('intent'),
            confidence=intent_data.get('confidence'),
            current_page=current_page,
            response_time_ms=response_time_ms
        )
        
        return jsonify({
            "response": response,
            "response_time_ms": response_time_ms
        })
        
    except Exception as e:
        logger.error(f"Admin chat error: {e}")
        return jsonify({"error": "Admin chat service temporarily unavailable"}), 500

def get_admin_analytics_data():
    """Get comprehensive admin analytics data"""
    if not mysql_db:
        return "Database connection unavailable"
    
    try:
        cursor = mysql_db.cursor(dictionary=True)
        analytics = {}
        
        # Sales Analytics - Fix table name from 'orders' to 'order'
        cursor.execute("""
            SELECT 
                COUNT(*) as total_orders,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_order_value,
                DATE(order_date) as order_date
            FROM `order` 
            WHERE order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(order_date)
            ORDER BY order_date DESC
            LIMIT 30
        """)
        analytics['daily_sales'] = cursor.fetchall()
        
        # Today's Performance - Fix table name
        cursor.execute("""
            SELECT 
                COUNT(*) as orders_today,
                COALESCE(SUM(total_amount), 0) as revenue_today,
                COALESCE(AVG(total_amount), 0) as avg_order_today
            FROM `order` 
            WHERE DATE(order_date) = CURDATE()
        """)
        analytics['today'] = cursor.fetchone()
        
        # Yesterday's Performance for comparison
        cursor.execute("""
            SELECT 
                COUNT(*) as orders_yesterday,
                COALESCE(SUM(total_amount), 0) as revenue_yesterday
            FROM `order` 
            WHERE DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
        """)
        analytics['yesterday'] = cursor.fetchone()
        
        # This Month's Performance
        cursor.execute("""
            SELECT 
                COUNT(*) as orders_this_month,
                COALESCE(SUM(total_amount), 0) as revenue_this_month,
                COALESCE(AVG(total_amount), 0) as avg_order_this_month
            FROM `order` 
            WHERE MONTH(order_date) = MONTH(CURRENT_DATE()) 
            AND YEAR(order_date) = YEAR(CURRENT_DATE())
        """)
        analytics['this_month'] = cursor.fetchone()
        
        # Top Products - Fix table names and add proper joins
        cursor.execute("""
            SELECT 
                p.product_name,
                p.product_price,
                p.product_type,
                SUM(oi.quantity) as units_sold,
                SUM(oi.total_price) as product_revenue,
                COUNT(DISTINCT oi.order_id) as order_count
            FROM order_items oi
            JOIN product p ON oi.product_id = p.product_id
            JOIN `order` o ON oi.order_id = o.order_id
            WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY p.product_id, p.product_name, p.product_price, p.product_type
            ORDER BY units_sold DESC
            LIMIT 10
        """)
        analytics['top_products'] = cursor.fetchall()
        
        # Low Stock Alerts with more details
        cursor.execute("""
            SELECT 
                p.product_name,
                p.product_type,
                p.product_price,
                i.stock_quantity,
                CASE 
                    WHEN i.stock_quantity = 0 THEN 'OUT OF STOCK'
                    WHEN i.stock_quantity <= 5 THEN 'CRITICAL'
                    WHEN i.stock_quantity <= 10 THEN 'LOW'
                    ELSE 'NORMAL'
                END as stock_status
            FROM inventory i
            JOIN product p ON i.product_id = p.product_id
            WHERE i.stock_quantity <= 15
            ORDER BY i.stock_quantity ASC
            LIMIT 20
        """)
        analytics['low_stock'] = cursor.fetchall()
        
        # Customer Analytics - Fix table name
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT c.customer_id) as active_customers,
                COUNT(DISTINCT o.order_id) as total_orders,
                COALESCE(AVG(o.total_amount), 0) as avg_customer_value,
                COUNT(DISTINCT CASE WHEN o.order_date >= CURDATE() THEN c.customer_id END) as customers_today
            FROM customer c
            LEFT JOIN `order` o ON c.customer_id = o.customer_id
            WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        """)
        analytics['customer_metrics'] = cursor.fetchone()
        
        # Total Customer Count
        cursor.execute("""
            SELECT COUNT(*) as total_customers
            FROM customer
        """)
        analytics['total_customers'] = cursor.fetchone()
        
        # Recent Messages/Inquiries with more context
        cursor.execute("""
            SELECT name, email, subject, message, created_at,
                   CASE 
                       WHEN created_at >= CURDATE() THEN 'Today'
                       WHEN created_at >= DATE_SUB(CURDATE(), INTERVAL 1 DAY) THEN 'Yesterday'
                       ELSE CONCAT(DATEDIFF(CURDATE(), created_at), ' days ago')
                   END as time_ago
            FROM messages 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        analytics['recent_inquiries'] = cursor.fetchall()
        
        # Payment Methods Analysis
        cursor.execute("""
            SELECT 
                payment_method,
                COUNT(*) as payment_count,
                SUM(payment_amount) as total_amount
            FROM payment p
            JOIN `order` o ON p.order_id = o.order_id
            WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY payment_method
            ORDER BY payment_count DESC
        """)
        analytics['payment_methods'] = cursor.fetchall()
        
        # Product Categories Performance
        cursor.execute("""
            SELECT 
                p.product_type as category,
                COUNT(DISTINCT p.product_id) as total_products,
                COALESCE(SUM(oi.quantity), 0) as units_sold,
                COALESCE(SUM(oi.total_price), 0) as category_revenue,
                AVG(p.product_price) as avg_price
            FROM product p
            LEFT JOIN order_items oi ON p.product_id = oi.product_id
            LEFT JOIN `order` o ON oi.order_id = o.order_id
            WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY) OR o.order_date IS NULL
            GROUP BY p.product_type
            ORDER BY category_revenue DESC
        """)
        analytics['category_performance'] = cursor.fetchall()
        
        cursor.close()
        return analytics
        
    except mysql.connector.Error as e:
        logger.error(f"Admin analytics query failed: {e}")
        return {"error": f"Analytics data unavailable: {str(e)}"}

class ImageSearchEngine:
    """Advanced image search with visual similarity matching using BLOB storage"""
    
    def __init__(self, processor, model, embedding_model):
        self.processor = processor
        self.model = model
        self.embedder = embedding_model
        
        # Product category mapping for database queries
        self.category_mapping = {
            'shirt': ['shirt', 'top', 'blouse', 't-shirt', 'polo', 'tank', 'vest'],
            'bag': ['bag', 'purse', 'handbag', 'backpack', 'tote', 'clutch'],
            'watch': ['watch', 'timepiece', 'smartwatch'],
            'shoes': ['shoes', 'sneakers', 'boots', 'sandals', 'heels'],
            'jeans': ['jeans', 'pants', 'trouser', 'denim'],
            'dress': ['dress', 'gown', 'frock'],
            'jacket': ['jacket', 'coat', 'blazer', 'hoodie'],
            'accessories': ['accessories', 'jewelry', 'necklace', 'bracelet']
        }
    
    def get_product_images_from_db(self, product_ids: List[int]) -> Dict[int, bytes]:
        """Retrieve product images from database BLOB storage for analysis"""
        if not mysql_db or not product_ids:
            return {}
        
        try:
            cursor = mysql_db.cursor(dictionary=True)
            placeholders = ','.join(['%s'] * len(product_ids))
            
            # First try product_images table
            query = f"""
            SELECT product_id, image_data 
            FROM product_images 
            WHERE product_id IN ({placeholders}) AND is_primary = true
            """
            cursor.execute(query, product_ids)
            results = cursor.fetchall()
            
            images = {}
            for row in results:
                if row['image_data']:
                    images[row['product_id']] = row['image_data']
            
            # Fallback to product table for missing images
            missing_ids = [pid for pid in product_ids if pid not in images]
            if missing_ids:
                placeholders = ','.join(['%s'] * len(missing_ids))
                query = f"""
                SELECT product_id, product_image_blob 
                FROM product 
                WHERE product_id IN ({placeholders}) AND product_image_blob IS NOT NULL
                """
                cursor.execute(query, missing_ids)
                fallback_results = cursor.fetchall()
                
                for row in fallback_results:
                    if row['product_image_blob']:
                        images[row['product_id']] = row['product_image_blob']
            
            cursor.close()
            return images
            
        except Exception as e:
            logger.error(f"Error retrieving product images: {e}")
            return {}
    
    def extract_image_features(self, image_bytes: bytes) -> Optional[List[float]]:
        """Extract image features using BLIP for similarity comparison"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Process with BLIP to get description
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50)
            description = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Create embedding from description for similarity
            embedding = self.embedder.encode(description)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def calculate_visual_similarity(self, uploaded_image_bytes: bytes, product_images: Dict[int, bytes]) -> Dict[int, float]:
        """Calculate visual similarity between uploaded image and product images"""
        similarities = {}
        
        try:
            # Extract features from uploaded image
            uploaded_features = self.extract_image_features(uploaded_image_bytes)
            if not uploaded_features:
                return similarities
            
            # Compare with each product image
            for product_id, image_bytes in product_images.items():
                product_features = self.extract_image_features(image_bytes)
                if product_features:
                    # Calculate cosine similarity
                    similarity = cosine_similarity([uploaded_features], [product_features])[0][0]
                    similarities[product_id] = float(similarity)
                    logger.info(f"Product {product_id} similarity: {similarity:.3f}")
            
            return similarities
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return similarities
    
    def find_similar_products(self, image_description: str, uploaded_image_bytes: bytes = None, limit: int = 6) -> List[Dict]:
        """Find products using advanced visual similarity matching"""
        if not mysql_db:
            return []
        
        try:
            # Detect product category first
            detected_category = self.detect_product_category(image_description)
            logger.info(f"Detected category: {detected_category}")
            
            cursor = mysql_db.cursor(dictionary=True)
            
            # Get all products with BLOB images for similarity comparison
            if uploaded_image_bytes:
                # Get products with images for visual matching
                query = """
                SELECT p.*, i.stock_quantity 
                FROM product p 
                LEFT JOIN inventory i ON p.product_id = i.product_id
                WHERE p.product_image_blob IS NOT NULL OR 
                      EXISTS (SELECT 1 FROM product_images pi WHERE pi.product_id = p.product_id AND pi.is_primary = true)
                ORDER BY p.product_id
                """
                cursor.execute(query)
                all_products = cursor.fetchall()
                
                if all_products:
                    # Get product images for similarity comparison
                    product_ids = [p['product_id'] for p in all_products]
                    product_images = self.get_product_images_from_db(product_ids)
                    
                    if product_images:
                        # Calculate visual similarities
                        similarities = self.calculate_visual_similarity(uploaded_image_bytes, product_images)
                        
                        # Filter products with good similarity (>0.3) or matching category
                        matched_products = []
                        category_products = []
                        
                        for product in all_products:
                            pid = product['product_id']
                            similarity = similarities.get(pid, 0)
                            
                            # Add similarity score to product
                            product['similarity_score'] = similarity
                            product['visual_match'] = similarity > 0.3
                            
                            # Check category match
                            product_category = product.get('product_type', '').lower()
                            category_match = detected_category in product_category or product_category in detected_category
                            
                            if similarity > 0.3:  # High visual similarity
                                product['match_reason'] = f"Visual similarity: {similarity:.1%}"
                                matched_products.append(product)
                            elif category_match:  # Category match
                                product['match_reason'] = f"Category match: {detected_category}"
                                category_products.append(product)
                        
                        # Sort by similarity and combine results
                        matched_products.sort(key=lambda x: x['similarity_score'], reverse=True)
                        category_products.sort(key=lambda x: x.get('product_ranking', 0), reverse=True)
                        
                        # Return best matches first, then category matches
                        final_products = matched_products[:3] + category_products[:3]
                        final_products = final_products[:limit]
                        
                        if matched_products:
                            logger.info(f"Found {len(matched_products)} visually similar products")
                        else:
                            logger.info(f"No visual matches, showing {len(category_products)} category matches")
                        
                        # Process for display
                        for product in final_products:
                            product['product_link'] = f"/sproductpage.html?id={product['product_id']}"
                            if product.get('stock_quantity') is None:
                                product['stock_quantity'] = 10
                            product['detected_category'] = detected_category
                        
                        return final_products
            
            # Fallback to category-based search if no visual matching
            if detected_category != "general":
                category_terms = self.category_mapping.get(detected_category, [detected_category])
                placeholders = ','.join(['%s'] * len(category_terms))
                query = f"""
                SELECT p.*, i.stock_quantity 
                FROM product p 
                LEFT JOIN inventory i ON p.product_id = i.product_id
                WHERE p.product_type IN ({placeholders})
                ORDER BY p.product_ranking DESC, p.product_id
                LIMIT %s
                """
                params = category_terms + [limit]
            else:
                query = """
                SELECT p.*, i.stock_quantity 
                FROM product p 
                LEFT JOIN inventory i ON p.product_id = i.product_id
                ORDER BY p.product_ranking DESC
                LIMIT %s
                """
                params = [limit]
            
            cursor.execute(query, params)
            products = cursor.fetchall()
            cursor.close()
            
            # Process products for display
            for product in products:
                product['product_link'] = f"/sproductpage.html?id={product['product_id']}"
                if product.get('stock_quantity') is None:
                    product['stock_quantity'] = 10
                product['detected_category'] = detected_category
                product['match_reason'] = f"Category: {detected_category}"
                product['similarity_score'] = 0
                product['visual_match'] = False
            
            return products
            
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return []
    
    def process_image(self, image_data: str) -> str:
        """Process image and return description"""
        try:
            # Decode base64 image
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Generate image caption
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            logger.info(f"Generated image description: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return "clothing item"
    
    def detect_product_category(self, description: str) -> str:
        """Detect the main product category from image description"""
        description_lower = description.lower()
        
        # Count matches for each category
        category_scores = {}
        for category, keywords in self.category_mapping.items():
            score = 0
            for keyword in keywords:
                if keyword in description_lower:
                    score += 1
            if score > 0:
                category_scores[category] = score
        
        # Return the category with highest score, or default to general search
        if category_scores:
            detected_category = max(category_scores, key=category_scores.get)
            logger.info(f"Detected category: {detected_category} (score: {category_scores[detected_category]})")
            return detected_category
        
        # Fallback: check for common clothing terms
        if any(word in description_lower for word in ['clothing', 'wear', 'garment', 'outfit']):
            return 'clothing'
        
        return 'general'

# Initialize Image Search Engine
    
    def extract_keywords(self, description: str) -> List[str]:
        """Extract relevant keywords from image description with enhanced recognition"""
        # Comprehensive fashion and clothing keywords
        fashion_keywords = {
            # Clothing types
            'shirt', 'dress', 'pants', 'jeans', 'jacket', 'coat', 'sweater', 'hoodie',
            'skirt', 'shorts', 'top', 'blouse', 'cardigan', 'blazer', 'vest', 'tshirt',
            'polo', 'tank', 'crop', 'maxi', 'mini', 'midi', 'formal', 'casual',
            'uniform', 'suit', 'trouser', 'leggings', 'jogger', 'sweatshirt',
            
            # Colors
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey', 'brown',
            'pink', 'purple', 'orange', 'navy', 'beige', 'khaki', 'denim', 'cream',
            'maroon', 'burgundy', 'teal', 'turquoise', 'lime', 'olive', 'gold', 'silver',
            
            # Materials
            'cotton', 'wool', 'silk', 'linen', 'polyester', 'leather', 'denim', 'velvet',
            'satin', 'chiffon', 'jersey', 'canvas', 'corduroy', 'fleece',
            
            # Patterns
            'striped', 'floral', 'solid', 'plaid', 'checkered', 'polka', 'geometric',
            'printed', 'embroidered', 'lace', 'paisley', 'animal', 'abstract',
            
            # Features
            'long', 'short', 'sleeve', 'sleeveless', 'collar', 'button', 'zip', 'zipper',
            'pocket', 'hood', 'belt', 'tie', 'bow', 'ruffle', 'pleated', 'fitted',
            'loose', 'tight', 'slim', 'regular', 'oversized',
            
            # Categories
            'men', 'women', 'unisex', 'boy', 'girl', 'adult', 'kids', 'child',
            'formal', 'casual', 'sport', 'athletic', 'business', 'party', 'wedding',
            
            # Accessories
            'hat', 'cap', 'scarf', 'gloves', 'bag', 'belt', 'shoes', 'sneakers',
            'boots', 'sandals', 'heels', 'flats', 'watch', 'jewelry', 'necklace',
            
            # Styles
            'vintage', 'modern', 'classic', 'trendy', 'bohemian', 'punk', 'gothic',
            'preppy', 'streetwear', 'minimalist', 'elegant', 'chic'
        }
        
        description_lower = description.lower()
        found_keywords = []
        
        # Direct keyword matching
        for keyword in fashion_keywords:
            if keyword in description_lower:
                found_keywords.append(keyword)
        
        # Additional smart extraction using word boundaries
        import re
        words = re.findall(r'\b\w+\b', description_lower)
        for word in words:
            if word in fashion_keywords and word not in found_keywords:
                found_keywords.append(word)
        
        # Extract potential brand names or specific descriptors
        for word in words:
            if len(word) > 3 and word not in found_keywords:
                # Check if it might be a relevant descriptor
                if any(fashion_word in word for fashion_word in ['wear', 'cloth', 'dress', 'style']):
                    found_keywords.append(word)
        
        return found_keywords[:8]  # Increased limit for better matching
    
    def calculate_advanced_similarity(self, description: str, product: Dict) -> float:
        """Calculate advanced similarity score using multiple factors"""
        total_score = 0.0
        
        # 1. Semantic similarity using embeddings
        semantic_score = self.calculate_semantic_similarity(description, product)
        total_score += semantic_score * 0.4  # 40% weight
        
        # 2. Keyword matching score
        keyword_score = self.calculate_keyword_similarity(description, product)
        total_score += keyword_score * 0.3  # 30% weight
        
        # 3. Category/type matching
        category_score = self.calculate_category_similarity(description, product)
        total_score += category_score * 0.2  # 20% weight
        
        # 4. Color and pattern matching
        visual_score = self.calculate_text_visual_similarity(description, product)
        total_score += visual_score * 0.1  # 10% weight
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def calculate_semantic_similarity(self, description: str, product: Dict) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        try:
            # Create product text
            product_text = f"{product.get('product_name', '')} {product.get('product_discription', '')} {product.get('product_type', '')}"
            
            if not product_text.strip():
                return 0.0
            
            # Generate embeddings
            desc_embedding = self.embedder.encode(description)
            product_embedding = self.embedder.encode(product_text)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                desc_embedding.reshape(1, -1), 
                product_embedding.reshape(1, -1)
            )[0][0]
            
            return max(0.0, float(similarity))  # Ensure non-negative and convert to Python float
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def calculate_keyword_similarity(self, description: str, product: Dict) -> float:
        """Calculate similarity based on keyword matching"""
        try:
            keywords = self.extract_keywords(description)
            if not keywords:
                return 0.0
            
            product_text = f"{product.get('product_name', '')} {product.get('product_discription', '')} {product.get('product_type', '')}".lower()
            
            matched_keywords = 0
            for keyword in keywords:
                if keyword.lower() in product_text:
                    matched_keywords += 1
            
            return matched_keywords / len(keywords) if keywords else 0.0
            
        except Exception as e:
            logger.error(f"Keyword similarity calculation failed: {e}")
            return 0.0
    
    def calculate_category_similarity(self, description: str, product: Dict) -> float:
        """Calculate similarity based on product category/type"""
        try:
            product_type = product.get('product_type', '').lower()
            description_lower = description.lower()
            
            if not product_type:
                return 0.0
            
            # Direct type match
            if product_type in description_lower:
                return 1.0
            
            # Category synonyms
            category_synonyms = {
                'shirt': ['top', 'blouse', 'tshirt', 'polo', 'dress shirt'],
                'dress': ['gown', 'frock', 'maxi', 'mini', 'midi'],
                'pants': ['trousers', 'jeans', 'leggings', 'slacks'],
                'jacket': ['coat', 'blazer', 'cardigan', 'hoodie'],
                'shoes': ['footwear', 'sneakers', 'boots', 'sandals', 'heels'],
                'bag': ['purse', 'handbag', 'backpack', 'tote'],
                'hat': ['cap', 'beanie', 'fedora']
            }
            
            for category, synonyms in category_synonyms.items():
                if product_type == category:
                    for synonym in synonyms:
                        if synonym in description_lower:
                            return 0.8
                elif product_type in synonyms and category in description_lower:
                    return 0.8
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Category similarity calculation failed: {e}")
            return 0.0
    
    def calculate_text_visual_similarity(self, description: str, product: Dict) -> float:
        """Calculate similarity based on visual features mentioned in text (color, pattern)"""
        try:
            description_lower = description.lower()
            product_text = f"{product.get('product_name', '')} {product.get('product_discription', '')}".lower()
            
            # Color matching
            colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'brown',
                     'pink', 'purple', 'orange', 'navy', 'beige', 'khaki']
            
            color_matches = 0
            for color in colors:
                if color in description_lower and color in product_text:
                    color_matches += 1
            
            # Pattern matching
            patterns = ['striped', 'floral', 'solid', 'plaid', 'checkered', 'polka', 'printed']
            pattern_matches = 0
            for pattern in patterns:
                if pattern in description_lower and pattern in product_text:
                    pattern_matches += 1
            
            # Style matching
            styles = ['casual', 'formal', 'vintage', 'modern', 'elegant', 'sporty']
            style_matches = 0
            for style in styles:
                if style in description_lower and style in product_text:
                    style_matches += 1
            
            total_visual_features = len(colors) + len(patterns) + len(styles)
            total_matches = color_matches + pattern_matches + style_matches
            
            return total_matches / total_visual_features if total_visual_features > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Visual similarity calculation failed: {e}")
            return 0.0
    
    def get_similarity_breakdown(self, description: str, product: Dict) -> Dict:
        """Get detailed breakdown of similarity factors for debugging"""
        return {
            'semantic': self.calculate_semantic_similarity(description, product),
            'keywords': self.calculate_keyword_similarity(description, product),
            'category': self.calculate_category_similarity(description, product),
            'visual': self.calculate_text_visual_similarity(description, product),
            'extracted_keywords': self.extract_keywords(description)
        }

# Initialize Image Search Engine
if image_processor and image_model:
    image_search = ImageSearchEngine(image_processor, image_model, embedding_model)
else:
    image_search = None

# ================================================================
# 16. API ENDPOINTS - IMAGE SEARCH FUNCTIONALITY
# ================================================================

@app.route('/api/image-search', methods=['POST'])
def image_search_endpoint():
    """
    Handle image-based product search with AI visual analysis.
    
    This endpoint:
    1. Accepts uploaded images from users
    2. Uses BLIP AI to generate image descriptions
    3. Detects product categories from images
    4. Performs visual similarity matching with database products
    5. Returns ranked product recommendations with similarity scores
    
    Expected JSON payload:
    {
        "image": "base64-encoded image data",
        "message": "optional user message about what they're looking for"
    }
    
    Returns:
        JSON response with matching products and similarity scores
    """
    try:
        # Check if image search functionality is available
        if not image_search:
            return jsonify({"error": "Image search not available"}), 503
        
        # Extract request data
        data = request.get_json()
        image_data = data.get('image')
        user_message = data.get('message', '')
        
        # Validate image data
        if not image_data:
            return jsonify({"error": "Image data required"}), 400
        
        # Process image using BLIP AI to get detailed description
        image_description = image_search.process_image(image_data)
        logger.info(f"🔍 Image description generated: {image_description}")
        
        # Detect product category using AI analysis
        detected_category = image_search.detect_product_category(image_description)
        logger.info(f"Detected category: {detected_category}")
        
        # Find products using visual similarity + category detection
        similar_products = image_search.find_similar_products(
            image_description, 
            uploaded_image_bytes=base64.b64decode(image_data.split(',')[1]) if image_data.startswith('data:image') else base64.b64decode(image_data), 
            limit=6
        )
        
        # Generate response message
        if similar_products:
            # Separate visual matches from category matches
            visual_matches = [p for p in similar_products if p.get('visual_match', False)]
            category_matches = [p for p in similar_products if not p.get('visual_match', False)]
            
            response_parts = [
                f"🔍 **Image Analysis:** {image_description}",
                f"📂 **Detected Category:** {detected_category.title()}",
                ""
            ]
            
            if visual_matches:
                response_parts.extend([
                    f"🎯 **Found {len(visual_matches)} Visually Similar Products:**",
                    ""
                ])
                
                for i, product in enumerate(visual_matches, 1):
                    stock_status = "✅ In Stock" if product.get('stock_quantity', 0) > 0 else "❌ Out of Stock"
                    similarity = product.get('similarity_score', 0)
                    
                    product_card = f"""**🔥 {i}. {product.get('product_name', 'Unknown')}** ⭐ {similarity:.1%} Match
💰 **Price:** ${product.get('product_price', 0):.2f} | 📂 **Type:** {product.get('product_type', 'N/A')} | {stock_status}
📝 **Description:** {product.get('product_discription', 'No description')[:100]}...
🛒 **[View & Buy Product](/sproductpage.html?id={product.get('product_id')})**
---"""
                    response_parts.append(product_card)
            
            if category_matches:
                if visual_matches:
                    response_parts.extend([
                        "",
                        f"📦 **More {detected_category.title()} Products:**",
                        ""
                    ])
                else:
                    response_parts.extend([
                        f"📦 **Found {len(category_matches)} {detected_category.title()} Products:**",
                        ""
                    ])
                
                # Create product cards for category matches
                for i, product in enumerate(category_matches, len(visual_matches) + 1):
                    stock_status = "✅ In Stock" if product.get('stock_quantity', 0) > 0 else "❌ Out of Stock"
                    
                    product_card = f"""**{i}. {product.get('product_name', 'Unknown')}**
💰 **Price:** ${product.get('product_price', 0):.2f} | 📂 **Type:** {product.get('product_type', 'N/A')} | {stock_status}
⭐ **Rating:** {'⭐' * (product.get('product_ranking', 1))}
📝 **Description:** {product.get('product_discription', 'No description')[:100]}...
� **[View & Buy Product](/sproductpage.html?id={product.get('product_id')})**
---"""
                    response_parts.append(product_card)
            
            # Add helpful tips and call-to-action
            response_parts.extend([
                "",
                "💡 **Tips:**",
                "• Click product links to see full details and add to cart",
                "• Products with higher match percentages are visually most similar", 
                "• All prices shown are current and stock levels are real-time",
                "",
                "💬 **Need Help?** Ask me about specific products, sizes, or styling advice!"
            ])
            response_text = "\n".join(response_parts)
            
            # Clean product data - remove BLOB fields that can't be JSON serialized
            clean_products = []
            for product in similar_products:
                clean_product = {
                    'product_id': product.get('product_id'),
                    'product_name': product.get('product_name'),
                    'product_price': product.get('product_price'),
                    'product_type': product.get('product_type'),
                    'product_discription': product.get('product_discription'),
                    'stock_quantity': product.get('stock_quantity'),
                    'product_link': product.get('product_link'),
                    'visual_similarity': product.get('visual_similarity', 0),
                    'visual_match': product.get('visual_match', False),
                    'match_reason': product.get('match_reason', 'Category match'),
                    'detected_category': detected_category,
                    'image_url': f"/api/product-image/{product.get('product_id')}"  # Use BLOB API endpoint
                    # Explicitly exclude: product_image_blob, image_data, etc.
                }
                clean_products.append(clean_product)
            
            return jsonify({
                "response": response_text,
                "image_description": image_description,
                "detected_category": detected_category,
                "products": clean_products, # Send product data for card display
                "total_found": len(similar_products),
                "visual_matches": len([p for p in similar_products if p.get('visual_match', False)]),
                "category_matches": len([p for p in similar_products if not p.get('visual_match', False)]),
                "display_mode": "complete_cards",
                "search_quality": "excellent" if any(p.get('visual_match', False) for p in similar_products) else "good"
            })
        else:
            return jsonify({
                "response": f"🔍 I can see this is a {detected_category}, but unfortunately, I couldn't find similar products in our current inventory. Would you like me to show you our latest {detected_category} collection instead?",
                "image_description": image_description,
                "detected_category": detected_category,
                "products": [],
                "total_found": 0
            })
            
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        return jsonify({"error": "Image search service temporarily unavailable"}), 500

# ================================================================
# 18. API ENDPOINTS - CHAT HISTORY MANAGEMENT
# ================================================================

@app.route('/api/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """
    Retrieve chat history for a specific session.
    
    Query Parameters:
    - limit: Number of messages to retrieve (default: 50)
    - offset: Number of messages to skip (default: 0)
    
    Returns:
        JSON response with chat history and session info
    """
    try:
        limit = int(request.args.get('limit', 50))
        
        # Get chat history from database
        history = chatbot.chat_history.get_session_history(session_id, limit)
        
        # Get session information
        session_info = {}
        if chatbot.db:
            cursor = chatbot.db.cursor(dictionary=True)
            cursor.execute("""
                SELECT session_id, user_email, session_type, first_message_at, 
                       last_message_at, total_messages, primary_intent
                FROM chat_sessions 
                WHERE session_id = %s
            """, (session_id,))
            session_info = cursor.fetchone() or {}
            cursor.close()
        
        return jsonify({
            "session_id": session_id,
            "session_info": session_info,
            "messages": history,
            "total_messages": len(history)
        })
        
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {e}")
        return jsonify({"error": "Failed to retrieve chat history"}), 500

@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """
    Get list of recent chat sessions for admin dashboard.
    
    Query Parameters:
    - limit: Number of sessions to retrieve (default: 20)
    - user_email: Filter by specific user email
    - session_type: Filter by session type ('customer' or 'admin')
    - days: Number of days to look back (default: 7)
    
    Returns:
        JSON response with session list and summary
    """
    try:
        limit = int(request.args.get('limit', 20))
        user_email = request.args.get('user_email')
        session_type = request.args.get('session_type')
        days = int(request.args.get('days', 7))
        
        if not chatbot.db:
            return jsonify({"error": "Database not available"}), 503
        
        cursor = chatbot.db.cursor(dictionary=True)
        
        # Build query with filters
        conditions = ["first_message_at >= DATE_SUB(NOW(), INTERVAL %s DAY)"]
        params = [days]
        
        if user_email:
            conditions.append("user_email = %s")
            params.append(user_email)
        
        if session_type:
            conditions.append("session_type = %s")
            params.append(session_type)
        
        query = f"""
            SELECT session_id, user_email, session_type, first_message_at, 
                   last_message_at, total_messages, primary_intent, resolution_status
            FROM chat_sessions 
            WHERE {' AND '.join(conditions)}
            ORDER BY last_message_at DESC 
            LIMIT %s
        """
        params.append(limit)
        
        cursor.execute(query, params)
        sessions = cursor.fetchall()
        
        # Get summary statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_sessions,
                COUNT(DISTINCT user_email) as unique_users,
                AVG(total_messages) as avg_messages_per_session,
                SUM(total_messages) as total_messages
            FROM chat_sessions 
            WHERE first_message_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        """, (days,))
        summary = cursor.fetchone()
        
        cursor.close()
        
        return jsonify({
            "sessions": sessions,
            "summary": summary,
            "filters": {
                "days": days,
                "user_email": user_email,
                "session_type": session_type,
                "limit": limit
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to retrieve chat sessions: {e}")
        return jsonify({"error": "Failed to retrieve chat sessions"}), 500

@app.route('/api/chat/analytics', methods=['GET'])
def get_chat_analytics():
    """
    Get chat analytics for admin dashboard.
    
    Query Parameters:
    - days: Number of days to analyze (default: 7)
    
    Returns:
        JSON response with comprehensive chat analytics
    """
    try:
        days = int(request.args.get('days', 7))
        
        # Get analytics from chat history manager
        analytics = chatbot.chat_history.get_chat_analytics(days)
        
        if not analytics:
            return jsonify({"error": "Analytics not available"}), 503
        
        return jsonify({
            "analytics": analytics,
            "period_days": days,
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get chat analytics: {e}")
        return jsonify({"error": "Failed to get chat analytics"}), 500

@app.route('/api/chat/user/<user_email>/sessions', methods=['GET'])
def get_user_chat_history(user_email):
    """
    Get chat session history for a specific user.
    
    Returns:
        JSON response with user's chat sessions and recent messages
    """
    try:
        # Get user's recent sessions
        sessions = chatbot.chat_history.get_user_recent_sessions(user_email, limit=10)
        
        # Get recent messages across all user sessions
        recent_messages = []
        if chatbot.db:
            cursor = chatbot.db.cursor(dictionary=True)
            cursor.execute("""
                SELECT session_id, sender, message, intent, created_at
                FROM chat_conversations 
                WHERE user_email = %s 
                ORDER BY created_at DESC 
                LIMIT 50
            """, (user_email,))
            recent_messages = cursor.fetchall()
            cursor.close()
        
        return jsonify({
            "user_email": user_email,
            "sessions": sessions,
            "recent_messages": recent_messages,
            "total_sessions": len(sessions)
        })
        
    except Exception as e:
        logger.error(f"Failed to get user chat history: {e}")
        return jsonify({"error": "Failed to get user chat history"}), 500

# ================================================================
# 19. API ENDPOINTS - HEALTH CHECK
# ================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with system status"""
    return jsonify({
        "status": "healthy",
        "mysql": "connected" if mysql_db else "disconnected", 
        "vector_db": "connected" if knowledge_collection else "disconnected",
        "timestamp": datetime.now().isoformat()
    })

# ================================================================
# 17. APPLICATION STARTUP AND MAIN EXECUTION
# ================================================================
if __name__ == '__main__':
    """
    Main application startup sequence.
    
    This section handles:
    1. Knowledge base initialization with vector embeddings
    2. Telegram bot integration setup
    3. Image search engine initialization
    4. Flask web server startup
    5. System status monitoring and logging
    """
    
    # Display startup banner with system information
    print("=" * 80)
    print("🚀 WEBSTORE AI CHATBOT BACKEND - STARTING UP")
    print("=" * 80)
    print("🔧 System Components:")
    print(f"   • Flask Web Server: Ready")
    print(f"   • Google Gemini AI: {'✅ Connected' if gemini_model else '❌ Failed'}")
    print(f"   • MySQL Database: {'✅ Connected' if mysql_db else '❌ Failed'}")
    print(f"   • ChromaDB Vector DB: {'✅ Connected' if knowledge_collection else '❌ Failed'}")
    print(f"   • Image Processing: {'✅ Ready' if image_processor else '❌ Failed'}")
    print("=" * 80)
    
    # Initialize comprehensive knowledge base on startup
    print("📚 Building AI knowledge base from database...")
    
    try:
        kb.build_knowledge_embeddings()
        print("✅ Knowledge base successfully initialized!")
        print("   • Product catalog embedded")
        print("   • Website features indexed")
        print("   • Business policies loaded")
        print("   • Customer service data processed")
    except Exception as e:
        print(f"⚠️ Knowledge base initialization failed: {e}")
        print("   • Chatbot will work with limited capabilities")
    
    # Initialize Telegram bot integration for mobile support
    print("🤖 Setting up Telegram bot integration...")
    try:
        telegram_bot = register_telegram_routes(app)
        print("✅ Telegram integration ready!")
        print("   • Mobile customers can chat via Telegram")
    except Exception as e:
        print(f"⚠️ Telegram integration failed: {e}")
        telegram_bot = None
    
    # Initialize advanced image search capabilities
    print("🖼️ Setting up image search engine...")
    try:
        if image_processor and image_model:
            image_search = ImageSearchEngine(mysql_db, image_processor, image_model)
            print("✅ Image search engine ready!")
            print("   • Visual similarity matching enabled")
            print("   • Category detection active")
        else:
            print("⚠️ Image search unavailable (missing AI models)")
    except Exception as e:
        print(f"⚠️ Image search initialization failed: {e}")
    
    # Final startup summary and server launch
    print("=" * 80)
    print("🌟 WEBSTORE AI CHATBOT - READY FOR CUSTOMERS!")
    print("=" * 80)
    print("📡 Server Endpoints Available:")
    print("   • Main Chat API: http://localhost:5000/api/chat")
    print("   • Admin Chat API: http://localhost:5000/api/admin/chat")
    print("   • Image Search API: http://localhost:5000/api/image-search")
    print("   • Health Check: http://localhost:5000/api/health")
    print("   • Telegram Webhook: http://localhost:5000/telegram/webhook")
    print("=" * 80)
    print("🚀 Starting Flask development server...")
    
    # Start the Flask web server
    app.run(host='0.0.0.0', port=5000, debug=False)

# ================================================================
# END OF WEBSTORE AI CHATBOT BACKEND
# ================================================================
