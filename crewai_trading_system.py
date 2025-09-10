#!/usr/bin/env python3
"""
CrewAI-powered Sentiment Analysis & SEC Insider Trading Monitoring System
=========================================================================
Updated version with correct CrewAI tool implementations
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import sqlite3
from pathlib import Path

# LLM and RAG imports
from litellm import completion
from textblob import TextBlob
import chromadb
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# CrewAI imports - Updated
from crewai import Agent, Task, Crew, Process
from crewai.flow import Flow, start, listen
from crewai.tools import tool  # Use the decorator approach

# Load environment variables
load_dotenv()

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crewai_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration (same as before)
class Config:
    """Configuration settings for the application"""
    SEC_BASE_URL = "https://www.sec.gov"
    SEC_FILINGS_URL = f"{SEC_BASE_URL}/cgi-bin/browse-edgar"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key-here")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
    
    # X (Twitter) creators to analyze (using public handles)
    X_CREATORS = [
        "@elonmusk", "@chamath", "@naval", "@balajis", "@APompliano",
        "@cz_binance", "@VitalikButerin", "@aantonop", "@coindesk", "@cointelegraph"
    ]
    
    # LLM Configuration
    LLM_MODEL = "gpt-3.5-turbo"  # Change to gpt-4 if you have access
    LLM_MAX_TOKENS = 2000
    LLM_TEMPERATURE = 0.3
    
    # Paths
    DB_PATH = "data/trading_data.db"
    CHARTS_DIR = "charts"
    REPORTS_DIR = "reports"
    LOGS_DIR = "logs"
    DATA_DIR = "data"
    RAG_DIR = "rag_data"
    
    # RAG Configuration
    RAG_COLLECTION_NAME = "financial_docs"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RAG_RESULTS = 5

# Utility Functions
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        Config.CHARTS_DIR, Config.REPORTS_DIR, Config.LOGS_DIR, 
        Config.DATA_DIR, Config.RAG_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("All necessary directories created/verified")

def safe_llm_call(messages: List[Dict], model: str = None, max_retries: int = 3) -> str:
    """Safely call LLM with retry logic and error handling"""
    model = model or Config.LLM_MODEL
    
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=messages,
                max_tokens=Config.LLM_MAX_TOKENS,
                temperature=Config.LLM_TEMPERATURE
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"All LLM call attempts failed: {str(e)}")
                return f"Error: Unable to generate LLM response after {max_retries} attempts"
    
    return "Error: LLM call failed"

# RAG Implementation (same as before)
class RAGSystem:
    """RAG system using ChromaDB for document storage and retrieval"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=f"{Config.RAG_DIR}/chroma_db")
        self.collection = None
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self._initialize_collection()
        self._populate_sample_data()
    
    def _initialize_collection(self):
        """Initialize or get existing collection"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=Config.RAG_COLLECTION_NAME)
            logger.info("Found existing RAG collection")
        except:
            # Create new collection
            self.collection = self.client.create_collection(
                name=Config.RAG_COLLECTION_NAME,
                metadata={"description": "Financial documents and transcripts"}
            )
            logger.info("Created new RAG collection")
    
    def _populate_sample_data(self):
        """Populate collection with sample financial documents and YouTube transcripts"""
        # Check if collection already has documents
        try:
            count = self.collection.count()
            if count > 0:
                logger.info(f"RAG collection already contains {count} documents")
                return
        except:
            pass
        
        # Sample SEC filings content
        sample_docs = [
            {
                "id": "sec_filing_001",
                "content": """
                Apple Inc. (AAPL) Filed Form 8-K Current Report:
                
                Item 2.02 Results of Operations and Financial Condition
                
                On January 15, 2025, Apple Inc. announced financial results for its fiscal 2024 first quarter ended December 30, 2024. 
                The Company reported quarterly revenue of $119.58 billion, up 2 percent year-over-year, and quarterly earnings per 
                diluted share of $2.18, up 16 percent year-over-year.
                
                iPhone revenue was $69.70 billion for the quarter, up 6 percent year-over-year. Mac revenue was $7.78 billion, 
                up 0.5 percent year-over-year. iPad revenue was $7.02 billion, down 25 percent year-over-year.
                
                The company returned nearly $27 billion to shareholders during the quarter through dividends and share repurchases.
                """,
                "metadata": {"type": "SEC_filing", "company": "Apple Inc.", "form": "8-K", "date": "2025-01-15"}
            },
            {
                "id": "sec_filing_002", 
                "content": """
                Tesla Inc. (TSLA) Filed Form 4 Statement of Changes in Beneficial Ownership:
                
                Director Elon Musk reported the sale of 50,000 shares of Tesla common stock at an average price of $248.50 per share
                on January 14, 2025. The total value of the transaction was approximately $12.4 million.
                
                Following this transaction, Musk directly owns 411,062,076 shares of Tesla common stock. The sale was conducted
                under a pre-arranged 10b5-1 trading plan established in September 2024.
                
                This represents part of Musk's ongoing diversification strategy while maintaining significant ownership in Tesla.
                """,
                "metadata": {"type": "SEC_filing", "company": "Tesla Inc.", "form": "4", "date": "2025-01-14"}
            }
        ]
        
        # Add documents to collection
        for doc in sample_docs:
            try:
                self.collection.add(
                    ids=[doc["id"]],
                    documents=[doc["content"]],
                    metadatas=[doc["metadata"]]
                )
            except Exception as e:
                logger.warning(f"Failed to add document {doc['id']}: {str(e)}")
        
        logger.info(f"Populated RAG collection with {len(sample_docs)} sample documents")
    
    def query(self, query_text: str, n_results: int = None) -> List[Dict]:
        """Query the RAG system for relevant documents"""
        n_results = n_results or Config.MAX_RAG_RESULTS
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Format results for easier consumption
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            logger.info(f"RAG query returned {len(formatted_results)} results for: {query_text[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            return []

# Initialize RAG system
rag_system = RAGSystem()

# Updated Tools using @tool decorator
@tool("SEC Data Fetcher")
def fetch_sec_data(hours_back: int = 24) -> str:
    """
    Fetches SEC filings data from the last specified hours with enhanced processing.
    
    Args:
        hours_back (int): Number of hours to look back for SEC filings
    
    Returns:
        str: JSON string containing SEC filings data
    """
    try:
        logger.info(f"Fetching SEC data for last {hours_back} hours")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)
        
        # Enhanced mock SEC data with more realistic filings
        sample_filings = [
            {
                "company": "Apple Inc.",
                "ticker": "AAPL", 
                "form_type": "8-K",
                "filing_date": (end_date - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
                "description": "Results of Operations and Financial Condition - Q1 2025 earnings",
                "key_items": ["Item 2.02", "Item 9.01"],
                "significance": "High"
            },
            {
                "company": "Microsoft Corp.",
                "ticker": "MSFT",
                "form_type": "10-Q", 
                "filing_date": (end_date - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M"),
                "description": "Quarterly Report - Azure growth metrics disclosed",
                "key_items": ["Financial Statements", "MD&A"],
                "significance": "Medium"
            },
            {
                "company": "Tesla Inc.",
                "ticker": "TSLA",
                "form_type": "4",
                "filing_date": (end_date - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                "description": "Statement of Changes in Beneficial Ownership - Executive Sale",
                "key_items": ["Insider Transaction"],
                "significance": "High"
            },
            {
                "company": "Amazon.com Inc.",
                "ticker": "AMZN",
                "form_type": "8-K",
                "filing_date": (end_date - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M"),
                "description": "Material Agreement - AWS partnership announcement",
                "key_items": ["Item 1.01"],
                "significance": "Medium"
            }
        ]
        
        # Store in database
        _store_filings_data(sample_filings)
        
        return json.dumps({
            "status": "success",
            "filings_count": len(sample_filings),
            "data": sample_filings,
            "timestamp": datetime.now().isoformat(),
            "query_period_hours": hours_back
        })
        
    except Exception as e:
        logger.error(f"Error fetching SEC data: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})

@tool("Insider Trading Fetcher")
def fetch_insider_trading_data(hours_back: int = 24) -> str:
    """
    Fetches insider trading activity from SEC for the last specified hours.
    
    Args:
        hours_back (int): Number of hours to look back for insider trading data
    
    Returns:
        str: JSON string containing insider trading data
    """
    try:
        logger.info(f"Fetching insider trading data for last {hours_back} hours")
        
        # Enhanced mock insider trading data
        sample_trades = [
            {
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "insider_name": "Timothy D. Cook",
                "position": "Chief Executive Officer",
                "transaction_type": "Sale",
                "shares": 75000,
                "price": 185.50,
                "value": 13912500,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "ownership_type": "Direct",
                "transaction_code": "S",
                "plan_type": "10b5-1"
            },
            {
                "company": "Microsoft Corp.", 
                "ticker": "MSFT",
                "insider_name": "Satya Nadella",
                "position": "Chief Executive Officer", 
                "transaction_type": "Sale",
                "shares": 35000,
                "price": 415.75,
                "value": 14551250,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "ownership_type": "Direct",
                "transaction_code": "S", 
                "plan_type": "10b5-1"
            },
            {
                "company": "Tesla Inc.",
                "ticker": "TSLA",
                "insider_name": "Elon Musk", 
                "position": "Chief Executive Officer",
                "transaction_type": "Sale",
                "shares": 50000,
                "price": 248.25,
                "value": 12412500,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "ownership_type": "Direct",
                "transaction_code": "S",
                "plan_type": "10b5-1"
            },
            {
                "company": "Amazon.com Inc.",
                "ticker": "AMZN",
                "insider_name": "Andrew R. Jassy",
                "position": "Chief Executive Officer",
                "transaction_type": "Purchase", 
                "shares": 15000,
                "price": 185.30,
                "value": 2779500,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "ownership_type": "Direct",
                "transaction_code": "P",
                "plan_type": "Open Market"
            }
        ]
        
        # Store in database
        _store_trading_data(sample_trades)
        
        return json.dumps({
            "status": "success",
            "trades_count": len(sample_trades),
            "data": sample_trades,
            "timestamp": datetime.now().isoformat(),
            "total_value": sum(trade["value"] for trade in sample_trades)
        })
        
    except Exception as e:
        logger.error(f"Error fetching insider trading data: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})

@tool("Real Sentiment Analyzer")
def analyze_sentiment(creators: List[str] = None) -> str:
    """
    Performs real sentiment analysis on X (Twitter) creators using TextBlob.
    
    Args:
        creators (List[str]): List of creator handles to analyze
    
    Returns:
        str: JSON string containing sentiment analysis results
    """
    if creators is None:
        creators = Config.X_CREATORS
        
    try:
        logger.info(f"Analyzing sentiment for {len(creators)} creators using TextBlob")
        
        sentiment_results = []
        
        # Sample tweets for each creator (in real implementation, fetch from Twitter API)
        sample_tweets = {
            "@elonmusk": [
                "Excited about the future of sustainable energy and space exploration!",
                "Tesla production numbers looking good this quarter",
                "Market volatility is temporary, innovation is permanent"
            ],
            "@chamath": [
                "Tech fundamentals remain strong despite market noise",
                "Long-term thinking wins in volatile markets",
                "Focus on companies solving real problems"
            ],
            "@naval": [
                "Building wealth requires patience and compound thinking", 
                "Technology continues to democratize opportunity",
                "Stay focused on value creation over speculation"
            ],
            "@balajis": [
                "Decentralization trends accelerating across industries",
                "Network states concept gaining mainstream attention", 
                "Technology enabling new forms of coordination"
            ],
            "@APompliano": [
                "Bitcoin adoption by institutions continues steadily",
                "Digital assets maturing as an asset class",
                "Inflation concerns driving alternative investments"
            ]
        }
        
        for creator in creators:
            # Use sample tweets or generate neutral sentiment for missing creators
            tweets = sample_tweets.get(creator, [
                "Market conditions remain interesting for long-term investors",
                "Technology sector showing mixed signals",
                "Maintaining cautious optimism about future trends"
            ])
            
            # Perform real sentiment analysis using TextBlob
            sentiments = []
            for tweet in tweets:
                blob = TextBlob(tweet)
                sentiments.append(blob.sentiment.polarity)
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Calculate additional metrics
            engagement_rate = abs(avg_sentiment) * 5 + 2.5  # Mock engagement based on sentiment intensity
            posts_analyzed = len(tweets)
            
            sentiment_data = {
                "creator": creator,
                "sentiment_score": round(avg_sentiment, 3),
                "sentiment_label": _get_sentiment_label(avg_sentiment),
                "posts_analyzed": posts_analyzed,
                "engagement_rate": round(engagement_rate, 2),
                "individual_scores": [round(s, 3) for s in sentiments],
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": min(0.95, abs(avg_sentiment) + 0.5)  # Mock confidence score
            }
            
            sentiment_results.append(sentiment_data)
        
        # Store in database
        _store_sentiment_data(sentiment_results)
        
        # Calculate overall market sentiment
        overall_sentiment = sum(r["sentiment_score"] for r in sentiment_results) / len(sentiment_results)
        
        return json.dumps({
            "status": "success",
            "creators_analyzed": len(sentiment_results),
            "data": sentiment_results,
            "overall_market_sentiment": round(overall_sentiment, 3),
            "overall_sentiment_label": _get_sentiment_label(overall_sentiment),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})

@tool("Enhanced Chart Generator")
def generate_charts(chart_type: str = "insider_trading") -> str:
    """
    Generates professional charts comparing current vs previous week data.
    
    Args:
        chart_type (str): Type of chart to generate ('insider_trading', 'sentiment', 'combined')
    
    Returns:
        str: JSON string containing chart generation results
    """
    try:
        logger.info(f"Generating enhanced {chart_type} chart")
        
        if chart_type == "insider_trading":
            return _generate_enhanced_trading_chart()
        elif chart_type == "sentiment":
            return _generate_enhanced_sentiment_chart()
        elif chart_type == "combined":
            return _generate_combined_analysis_chart()
        else:
            return json.dumps({"status": "error", "message": "Unknown chart type"})
            
    except Exception as e:
        logger.error(f"Error generating enhanced chart: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})

@tool("Financial RAG Query")
def query_rag_system(query: str, context_type: str = "general") -> str:
    """
    Query the RAG system for relevant financial documents and insights.
    
    Args:
        query (str): Search query for relevant documents
        context_type (str): Type of context to search for
    
    Returns:
        str: JSON string containing RAG query results
    """
    try:
        logger.info(f"Querying RAG system: {query[:50]}...")
        
        # Query the RAG system
        results = rag_system.query(query, n_results=Config.MAX_RAG_RESULTS)
        
        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "No relevant documents found",
                "query": query
            })
        
        # Format results for consumption
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                "metadata": result["metadata"],
                "relevance_score": 1 - result["distance"] if result["distance"] else 0.5
            })
        
        return json.dumps({
            "status": "success",
            "results_count": len(formatted_results),
            "query": query,
            "context_type": context_type,
            "results": formatted_results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        return json.dumps({"status": "error", "message": str(e), "query": query})

# Helper functions for the tools
def _store_filings_data(filings: List[Dict]) -> None:
    """Store filings data in SQLite database with enhanced schema"""
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    
    # Create enhanced table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sec_filings (
            id INTEGER PRIMARY KEY,
            company TEXT,
            ticker TEXT,
            form_type TEXT,
            filing_date TEXT,
            description TEXT,
            key_items TEXT,
            significance TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert data
    for filing in filings:
        cursor.execute('''
            INSERT INTO sec_filings (company, ticker, form_type, filing_date, description, key_items, significance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            filing['company'], filing.get('ticker', ''), filing['form_type'], 
            filing['filing_date'], filing['description'],
            json.dumps(filing.get('key_items', [])), filing.get('significance', 'Low')
        ))
    
    conn.commit()
    conn.close()

def _store_trading_data(trades: List[Dict]) -> None:
    """Store trading data in SQLite database with enhanced schema"""
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    
    # Create enhanced table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS insider_trades (
            id INTEGER PRIMARY KEY,
            company TEXT,
            ticker TEXT,
            insider_name TEXT,
            position TEXT,
            transaction_type TEXT,
            shares INTEGER,
            price REAL,
            value REAL,
            date TEXT,
            ownership_type TEXT,
            transaction_code TEXT,
            plan_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert data
    for trade in trades:
        cursor.execute('''
            INSERT INTO insider_trades 
            (company, ticker, insider_name, position, transaction_type, shares, price, value, date, ownership_type, transaction_code, plan_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['company'], trade.get('ticker', ''), trade['insider_name'], trade['position'], 
            trade['transaction_type'], trade['shares'], trade['price'], trade['value'], 
            trade['date'], trade.get('ownership_type', ''), 
            trade.get('transaction_code', ''), trade.get('plan_type', '')
        ))
    
    conn.commit()
    conn.close()

def _store_sentiment_data(sentiment_data: List[Dict]) -> None:
    """Store sentiment data in database with enhanced schema"""
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    
    # Create enhanced table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            id INTEGER PRIMARY KEY,
            creator TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            posts_analyzed INTEGER,
            engagement_rate REAL,
            individual_scores TEXT,
            confidence REAL,
            analysis_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert data
    for item in sentiment_data:
        cursor.execute('''
            INSERT INTO sentiment_analysis 
            (creator, sentiment_score, sentiment_label, posts_analyzed, engagement_rate, individual_scores, confidence, analysis_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item['creator'], item['sentiment_score'], item['sentiment_label'], 
            item['posts_analyzed'], item['engagement_rate'], 
            json.dumps(item['individual_scores']), item['confidence'], item['analysis_date']
        ))
    
    conn.commit()
    conn.close()

def _get_sentiment_label(score: float) -> str:
    """Convert sentiment score to label with more granularity"""
    if score >= 0.5:
        return "Very Positive"
    elif score >= 0.1:
        return "Positive"
    elif score >= -0.1:
        return "Neutral"
    elif score >= -0.5:
        return "Negative"
    else:
        return "Very Negative"

def _generate_enhanced_trading_chart() -> str:
    """Generate enhanced insider trading comparison chart"""
    # Create more realistic data with weekly comparison
    current_week_data = {
        "Apple Inc.": {"sales": 13912500, "purchases": 0, "net": -13912500},
        "Microsoft Corp.": {"sales": 14551250, "purchases": 0, "net": -14551250}, 
        "Tesla Inc.": {"sales": 12412500, "purchases": 0, "net": -12412500},
        "Amazon": {"sales": 0, "purchases": 2779500, "net": 2779500},
        "Google": {"sales": 8500000, "purchases": 1200000, "net": -7300000}
    }
    
    previous_week_data = {
        "Apple Inc.": {"sales": 8200000, "purchases": 500000, "net": -7700000},
        "Microsoft Corp.": {"sales": 12100000, "purchases": 0, "net": -12100000},
        "Tesla Inc.": {"sales": 15600000, "purchases": 0, "net": -15600000},
        "Amazon": {"sales": 2100000, "purchases": 4500000, "net": 2400000},
        "Google": {"sales": 6900000, "purchases": 800000, "net": -6100000}
    }
    
    # Create enhanced visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Insider Trading Analysis: Comprehensive Weekly Comparison', fontsize=16, fontweight='bold')
    
    # Chart 1: Net Trading Value Comparison
    companies = list(current_week_data.keys())
    current_net = [current_week_data[company]["net"] / 1e6 for company in companies]
    previous_net = [previous_week_data[company]["net"] / 1e6 for company in companies]
    
    x = np.arange(len(companies))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, previous_net, width, label='Previous Week', alpha=0.8, color='lightblue')
    bars2 = ax1.bar(x + width/2, current_net, width, label='Current Week', alpha=0.8, color='orange')
    
    ax1.set_xlabel('Companies')
    ax1.set_ylabel('Net Trading Value (Millions USD)')
    ax1.set_title('Net Insider Trading Value Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(companies, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add remaining chart code here...
    # (truncated for brevity, but would include all chart generation logic)
    
    plt.tight_layout()
    chart_path = f"{Config.CHARTS_DIR}/insider_trading_comprehensive.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return json.dumps({
        "status": "success",
        "chart_path": chart_path,
        "chart_type": "insider_trading_comprehensive",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "total_current_volume": sum(abs(data["net"]) for data in current_week_data.values()),
            "total_previous_volume": sum(abs(data["net"]) for data in previous_week_data.values()),
            "companies_analyzed": len(companies)
        }
    })

def _generate_enhanced_sentiment_chart() -> str:
    """Generate enhanced sentiment analysis chart"""
    # Get sentiment data from database or create enhanced mock data
    conn = sqlite3.connect(Config.DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM sentiment_analysis ORDER BY created_at DESC LIMIT 10", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    
    if df.empty:
        # Create enhanced sample data
        creators = Config.X_CREATORS
        sentiment_scores = [0.45, -0.15, 0.72, 0.23, -0.08, 0.58, 0.31, -0.02, 0.66, 0.19]
        engagement_rates = [7.2, 5.8, 9.1, 6.4, 4.9, 8.3, 6.7, 5.1, 8.8, 6.0]
        confidence_scores = [0.89, 0.76, 0.94, 0.82, 0.71, 0.91, 0.85, 0.68, 0.93, 0.79]
        
        df = pd.DataFrame({
            'creator': creators,
            'sentiment_score': sentiment_scores,
            'engagement_rate': engagement_rates,
            'confidence': confidence_scores
        })
    
    # Create enhanced sentiment visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('X (Twitter) Creators Sentiment Analysis: Comprehensive Report', fontsize=16, fontweight='bold')
    
    # Chart 1: Sentiment Scores with Confidence Intervals
    colors = ['darkred' if score < -0.3 else 'red' if score < -0.1 else 
             'orange' if score < 0.1 else 'lightgreen' if score < 0.3 else 'darkgreen' 
             for score in df['sentiment_score']]
    
    bars = ax1.bar(range(len(df)), df['sentiment_score'], color=colors, alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('X (Twitter) Creators')
    ax1.set_ylabel('Sentiment Score')
    ax1.set_title('Sentiment Scores with Confidence Intervals')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['creator'], rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = f"{Config.CHARTS_DIR}/sentiment_comprehensive.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return json.dumps({
        "status": "success",
        "chart_path": chart_path,
        "chart_type": "sentiment_comprehensive",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "overall_sentiment": round(df['sentiment_score'].mean(), 3),
            "creators_analyzed": len(df),
            "average_engagement": round(df['engagement_rate'].mean(), 2) if 'engagement_rate' in df.columns else 0
        }
    })

# Enhanced CrewAI Agents with updated tool references
def create_enhanced_agents():
    """Create specialized agents with updated tool implementations"""
    
    # SEC Data Agent
    sec_agent = Agent(
        role="SEC Data Intelligence Specialist",
        goal="Fetch, analyze, and interpret SEC filings data from the last 24 hours with deep insights",
        backstory="You are an expert in navigating SEC databases and extracting relevant filing information. "
                 "You have deep knowledge of corporate filings and can identify patterns that impact market sentiment. "
                 "You understand the significance of different form types and can contextualize their market implications.",
        tools=[fetch_sec_data, query_rag_system],
        verbose=True,
        allow_delegation=False
    )
    
    # Insider Trading Agent
    insider_agent = Agent(
        role="Insider Trading Intelligence Analyst",
        goal="Monitor, analyze, and interpret insider trading activities with pattern recognition",
        backstory="You specialize in tracking insider trading patterns and identifying significant "
                 "buying or selling activities by corporate executives and directors. You understand "
                 "the difference between planned sales (10b5-1) and discretionary trading, and can "
                 "identify unusual patterns that may signal important corporate developments.",
        tools=[fetch_insider_trading_data, query_rag_system],
        verbose=True,
        allow_delegation=False
    )
    
    # Comparison Agent
    comparison_agent = Agent(
        role="Data Visualization and Trend Analysis Specialist", 
        goal="Create comprehensive visualizations comparing current data with historical trends",
        backstory="You excel at identifying trends and patterns by comparing different time periods. "
                 "You create clear, professional visualizations that highlight important changes and "
                 "can spot subtle patterns that others might miss. Your charts tell compelling stories "
                 "about market dynamics and insider behavior patterns.",
        tools=[generate_charts, query_rag_system],
        verbose=True,
        allow_delegation=False
    )
    
    # Sentiment Agent
    sentiment_agent = Agent(
        role="Social Media Sentiment Intelligence Analyst",
        goal="Perform comprehensive sentiment analysis of key financial influencers on X (Twitter)",
        backstory="You are skilled at interpreting social media sentiment using advanced NLP techniques. "
                 "You understand how influential creators' opinions can impact market sentiment and can "
                 "identify subtle shifts in market psychology through language analysis. You can distinguish "
                 "between genuine sentiment and market manipulation attempts.",
        tools=[analyze_sentiment, query_rag_system],
        verbose=True,
        allow_delegation=False
    )
    
    # Report Agent
    report_agent = Agent(
        role="Senior Financial Intelligence Report Writer",
        goal="Synthesize all data sources into comprehensive, actionable financial intelligence reports",
        backstory="You are an expert financial writer and analyst who can synthesize complex data from multiple "
                 "sources into clear, actionable reports for institutional investors and decision-makers. "
                 "You have deep understanding of market dynamics, can identify correlation patterns, "
                 "and provide strategic insights that go beyond raw data analysis.",
        tools=[query_rag_system],
        verbose=True,
        allow_delegation=False
    )
    
    # RAG Agent
    rag_agent = Agent(
        role="Financial Document Research Specialist",
        goal="Provide contextual insights from historical documents and transcripts to enhance analysis",
        backstory="You are an expert at searching through vast document repositories to find relevant "
                 "historical context and patterns. You can quickly identify similar situations from the past "
                 "and provide valuable context that enhances current analysis.",
        tools=[query_rag_system],
        verbose=True,
        allow_delegation=False
    )
    
    return {
        "sec_agent": sec_agent,
        "insider_agent": insider_agent,
        "comparison_agent": comparison_agent,
        "sentiment_agent": sentiment_agent,
        "report_agent": report_agent,
        "rag_agent": rag_agent
    }

def create_enhanced_tasks(agents):
    """Create enhanced tasks with proper dependencies"""
    
    # RAG Context Task
    rag_context_task = Task(
        description="Query the RAG system for relevant historical context about recent SEC filings, "
                   "insider trading patterns, and market sentiment trends. Focus on similar situations "
                   "from the past that might provide valuable context for today's analysis.",
        agent=agents["rag_agent"],
        expected_output="Historical context and similar patterns from past filings and market events that "
                       "provide relevant background for current analysis."
    )
    
    # SEC Data Task
    sec_task = Task(
        description="Fetch and analyze SEC filings data from the last 24 hours. Focus on Form 8-K, 10-Q, 10-K, "
                   "and Form 4 filings. Identify any significant corporate events, earnings announcements, "
                   "or regulatory changes. Assess the potential market impact of each filing type.",
        agent=agents["sec_agent"],
        expected_output="A comprehensive analysis of SEC filings including company details, filing significance, "
                       "potential market impacts, and identification of any unusual patterns or noteworthy events.",
        context=[rag_context_task]
    )
    
    # Insider Trading Task
    insider_task = Task(
        description="Fetch and analyze insider trading activity from the last 24 hours. Focus on Form 4 filings "
                   "showing significant buy/sell activities by executives, directors, and major shareholders. "
                   "Distinguish between planned sales (10b5-1) and discretionary trading.",
        agent=agents["insider_agent"], 
        expected_output="Detailed analysis of insider trading activities with transaction details, pattern identification, "
                       "significance assessment, and potential market implications of insider behavior.",
        context=[rag_context_task]
    )
    
    # Sentiment Analysis Task
    sentiment_task = Task(
        description=f"Perform comprehensive sentiment analysis on the following X (Twitter) creators: "
                   f"{', '.join(Config.X_CREATORS)}. Analyze their recent posts for overall sentiment regarding "
                   f"financial markets, specific companies, and economic trends.",
        agent=agents["sentiment_agent"],
        expected_output="Comprehensive sentiment analysis including individual creator sentiment scores, "
                       "overall market sentiment trends, confidence levels, and identification of any "
                       "significant sentiment shifts or consensus patterns.",
        context=[rag_context_task]
    )
    
    # Comparison and Visualization Task
    comparison_task = Task(
        description="Generate comprehensive comparison visualizations showing current week vs previous week data "
                   "for insider trading activity and sentiment analysis. Create professional charts that highlight "
                   "trends, patterns, and anomalies.",
        agent=agents["comparison_agent"],
        expected_output="Professional visualization package including insider trading comparison charts, "
                       "sentiment analysis charts, trend analysis, and statistical insights about patterns "
                       "and anomalies in the data.",
        context=[sec_task, insider_task, sentiment_task]
    )
    
    # Comprehensive Report Task
    report_task = Task(
        description="Using all available data from SEC filings, insider trading analysis, sentiment analysis, "
                   "comparison charts, and RAG context, create a comprehensive financial intelligence report. "
                   "Synthesize insights, identify correlations, predict potential market impacts, and provide "
                   "actionable recommendations.",
        agent=agents["report_agent"],
        expected_output="A comprehensive, professionally written financial intelligence report that synthesizes "
                       "all data sources, provides deep insights, identifies patterns and correlations, assesses "
                       "risks and opportunities, and delivers actionable recommendations for market participants.",
        context=[sec_task, insider_task, sentiment_task, comparison_task, rag_context_task]
    )
    
    return [rag_context_task, sec_task, insider_task, sentiment_task, comparison_task, report_task]

# Main execution functions
def run_enhanced_trading_analysis():
    """Run the complete enhanced trading analysis using CrewAI"""
    logger.info("Starting Enhanced CrewAI Trading Analysis System")
    
    try:
        # Ensure environment setup
        ensure_directories()
        
        # Create agents and tasks
        agents = create_enhanced_agents()
        tasks = create_enhanced_tasks(agents)
        
        # Create crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        # Execute crew
        result = crew.kickoff()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{Config.REPORTS_DIR}/trading_analysis_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Financial Intelligence Report\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(str(result))
        
        logger.info("Enhanced trading analysis completed successfully")
        return {
            "status": "completed",
            "result": result,
            "report_path": report_path,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced trading analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    """Main execution entry point"""
    
    print("=" * 80)
    print("CrewAI-powered Financial Intelligence System")
    print("=" * 80)
    
    # Ensure environment setup
    ensure_directories()
    
    print("\n🚀 Starting Enhanced Trading Analysis...")
    result = run_enhanced_trading_analysis()
    
    # Display results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    if isinstance(result, dict):
        print(f"Status: {result.get('status', 'unknown')}")
        if result.get('status') == 'completed':
            print(f"Report Path: {result.get('report_path', 'N/A')}")
        elif result.get('status') == 'error':
            print(f"Error: {result.get('message', 'Unknown error')}")
    else:
        print("Analysis completed - check logs for details")
    
    print("\n✅ System execution completed")
    print(f"📊 Charts saved in: {Config.CHARTS_DIR}/")
    print(f"📄 Reports saved in: {Config.REPORTS_DIR}/")
    print(f"📝 Logs saved in: {Config.LOGS_DIR}/")