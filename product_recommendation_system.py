

import json
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ProductRecommendationSystem:
    def __init__(self):
        # Load pre-trained sentence transformer model for semantic understanding
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Product database with categories and descriptions
        self.product_database = {
            # Communication & Technology
            "smartphone": {
                "category": "technology",
                "description": "mobile phone communication device contact calls messages",
                "keywords": ["phone", "call", "contact", "number", "message", "text"]
            },
            "contact_management_app": {
                "category": "technology", 
                "description": "contact organizer phone book address management",
                "keywords": ["contact", "phone", "number", "address", "organize"]
            },
            
            # Entertainment
            "netflix_subscription": {
                "category": "entertainment",
                "description": "streaming service movies shows entertainment watch",
                "keywords": ["netflix", "watch", "movie", "show", "entertainment", "streaming"]
            },
            "youtube_premium": {
                "category": "entertainment",
                "description": "video streaming platform stand-up comedy entertainment",
                "keywords": ["youtube", "video", "comedy", "stand-up", "entertainment"]
            },
            "concert_tickets": {
                "category": "entertainment",
                "description": "live music performance entertainment event tickets",
                "keywords": ["music", "concert", "performance", "entertainment", "event"]
            },
            
            # Fashion & Clothing
            "quality_trousers": {
                "category": "fashion",
                "description": "high-quality pants clothing fashion outfit",
                "keywords": ["trousers", "pants", "clothing", "outfit", "fashion", "quality"]
            },
            "fashion_consulting_service": {
                "category": "fashion",
                "description": "style advice fashion consultation outfit planning",
                "keywords": ["fashion", "style", "outfit", "advice", "clothing", "consultation"]
            },
            "winter_scarf": {
                "category": "fashion",
                "description": "warm winter accessory health protection cold weather",
                "keywords": ["scarf", "winter", "cold", "warm", "health", "accessory"]
            },
            "online_shopping_platforms": {
                "category": "fashion",
                "description": "e-commerce fashion websites clothing shopping online",
                "keywords": ["shopping", "online", "website", "clothes", "fashion", "buy"]
            },
            
            # Books & Education
            "book_subscription_service": {
                "category": "books",
                "description": "book reading service literature education knowledge",
                "keywords": ["books", "reading", "literature", "education", "knowledge"]
            },
            "e_reader_device": {
                "category": "books",
                "description": "electronic reading device books digital library",
                "keywords": ["reading", "books", "e-reader", "digital", "library"]
            },
            
            # Health & Wellness
            "energy_supplements": {
                "category": "health",
                "description": "vitamins energy boost fatigue tiredness work performance",
                "keywords": ["tired", "energy", "fatigue", "work", "performance", "boost"]
            },
            "pain_relief_medication": {
                "category": "health",
                "description": "headache pain relief medication healthcare pharmacy",
                "keywords": ["headache", "pain", "relief", "medication", "health"]
            },
            "ergonomic_office_chair": {
                "category": "health",
                "description": "comfortable office furniture work productivity health",
                "keywords": ["work", "office", "chair", "comfortable", "ergonomic", "productivity"]
            },
            
            # Food & Dining
            "meal_delivery_service": {
                "category": "food",
                "description": "food delivery restaurant meals cooking dinner",
                "keywords": ["dinner", "food", "meal", "delivery", "restaurant", "cooking"]
            },
            "italian_restaurant_guide": {
                "category": "food",
                "description": "dining guide restaurant recommendations italian cuisine",
                "keywords": ["restaurant", "italian", "dining", "food", "cuisine"]
            },
            
            # Transportation
            "car_rental_service": {
                "category": "transportation",
                "description": "vehicle rental transportation travel car hire",
                "keywords": ["car", "rental", "travel", "transportation", "vehicle"]
            },
            "public_transport_app": {
                "category": "transportation",
                "description": "train bus transportation schedule travel planning",
                "keywords": ["train", "transport", "travel", "schedule", "bus"]
            },
            
            # Home & Office
            "paper_shredder": {
                "category": "office",
                "description": "office equipment document security paper destruction",
                "keywords": ["paper", "shredder", "office", "document", "security"]
            },
            "home_security_system": {
                "category": "home",
                "description": "apartment security safety home protection",
                "keywords": ["apartment", "security", "home", "safety", "protection"]
            },
            
            # Party & Events
            "party_planning_service": {
                "category": "events",
                "description": "party organization celebration event planning birthday",
                "keywords": ["party", "celebration", "birthday", "event", "planning"]
            },
            "party_supplies": {
                "category": "events",
                "description": "celebration decorations party accessories supplies",
                "keywords": ["party", "supplies", "decorations", "celebration", "accessories"]
            },
            
            # Beauty & Personal Care
            "beauty_salon_services": {
                "category": "beauty",
                "description": "beauty therapy massage salon treatments personal care",
                "keywords": ["beauty", "salon", "massage", "treatment", "therapy"]
            },
            
            # Travel
            "travel_insurance": {
                "category": "travel",
                "description": "vacation trip insurance travel protection coverage",
                "keywords": ["vacation", "travel", "trip", "insurance", "protection"]
            },
            "weekend_getaway_packages": {
                "category": "travel",
                "description": "short vacation weekend trip travel package relaxation",
                "keywords": ["weekend", "vacation", "trip", "getaway", "travel"]
            },
            
            # Mental Health & Support
            "therapy_consultation": {
                "category": "mental_health",
                "description": "psychological support depression counseling mental health",
                "keywords": ["depression", "therapy", "mental", "health", "counseling", "support"]
            },
            "meditation_app": {
                "category": "mental_health",
                "description": "mindfulness meditation stress relief mental wellness",
                "keywords": ["stress", "meditation", "mindfulness", "wellness", "mental"]
            },
            
            # Finance
            "budgeting_app": {
                "category": "finance",
                "description": "money management budget financial planning expenses",
                "keywords": ["money", "budget", "financial", "expenses", "planning"]
            }
        }
        
        # Create embeddings for product descriptions
        self.product_embeddings = {}
        for product, info in self.product_database.items():
            text = f"{info['description']} {' '.join(info['keywords'])}"
            self.product_embeddings[product] = self.model.encode(text)
    
    def extract_keywords(self, summary: str) -> List[str]:
        """Extract relevant keywords from summary"""
        # Common words to ignore
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
        
        # Clean and split text
        words = re.findall(r'\w+', summary.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def analyze_summary_content(self, summary: str) -> Dict[str, float]:
        """Analyze summary content and categorize by themes"""
        categories = {
            'communication': 0,
            'entertainment': 0,
            'fashion': 0,
            'health': 0,
            'food': 0,
            'travel': 0,
            'work': 0,
            'shopping': 0,
            'social': 0,
            'technology': 0
        }
        
        summary_lower = summary.lower()
        
        # Communication indicators
        comm_words = ['call', 'phone', 'contact', 'text', 'message', 'number']
        categories['communication'] = sum(1 for word in comm_words if word in summary_lower)
        
        # Entertainment indicators
        ent_words = ['movie', 'watch', 'netflix', 'youtube', 'comedy', 'show', 'entertainment']
        categories['entertainment'] = sum(1 for word in ent_words if word in summary_lower)
        
        # Fashion indicators
        fashion_words = ['clothes', 'trousers', 'scarf', 'shopping', 'outfit', 'fashion']
        categories['fashion'] = sum(1 for word in fashion_words if word in summary_lower)
        
        # Health indicators
        health_words = ['tired', 'headache', 'pain', 'health', 'cold', 'sick']
        categories['health'] = sum(1 for word in health_words if word in summary_lower)
        
        # Food indicators
        food_words = ['dinner', 'food', 'restaurant', 'meal', 'cooking', 'eat']
        categories['food'] = sum(1 for word in food_words if word in summary_lower)
        
        # Travel indicators
        travel_words = ['travel', 'vacation', 'trip', 'car', 'train', 'weekend']
        categories['travel'] = sum(1 for word in travel_words if word in summary_lower)
        
        # Work indicators
        work_words = ['work', 'office', 'job', 'boss', 'tired', 'bored']
        categories['work'] = sum(1 for word in work_words if word in summary_lower)
        
        # Shopping indicators
        shop_words = ['buy', 'shopping', 'store', 'purchase', 'money']
        categories['shopping'] = sum(1 for word in shop_words if word in summary_lower)
        
        # Social indicators
        social_words = ['party', 'birthday', 'friends', 'celebration', 'wedding']
        categories['social'] = sum(1 for word in social_words if word in summary_lower)
        
        # Technology indicators
        tech_words = ['phone', 'app', 'online', 'website', 'digital']
        categories['technology'] = sum(1 for word in tech_words if word in summary_lower)
        
        return categories
    
    def get_recommendations(self, summary: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Get product recommendations for a given summary"""
        # Create embedding for the summary
        summary_embedding = self.model.encode(summary)
        
        # Calculate similarities with all products
        similarities = {}
        for product, product_embedding in self.product_embeddings.items():
            similarity = cosine_similarity([summary_embedding], [product_embedding])[0][0]
            similarities[product] = similarity
        
        # Get content analysis
        content_analysis = self.analyze_summary_content(summary)
        
        # Boost scores based on content analysis
        for product, info in self.product_database.items():
            category = info['category']
            if category in content_analysis and content_analysis[category] > 0:
                similarities[product] *= (1 + content_analysis[category] * 0.2)
            
            # Keyword matching boost
            keywords = self.extract_keywords(summary)
            keyword_matches = sum(1 for keyword in keywords if keyword in info['keywords'])
            if keyword_matches > 0:
                similarities[product] *= (1 + keyword_matches * 0.1)
        
        # Sort by similarity and return top recommendations
        sorted_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for product, score in sorted_products[:top_k]:
            category = self.product_database[product]['category']
            recommendations.append((product, score, category))
        
        return recommendations
    
    def format_product_name(self, product_key: str) -> str:
        """Format product key into readable name"""
        return product_key.replace('_', ' ').title()
    
    def get_recommendation_explanation(self, summary: str, product: str) -> str:
        """Generate explanation for why a product was recommended"""
        summary_lower = summary.lower()
        product_info = self.product_database[product]
        
        # Find matching keywords
        matching_keywords = [kw for kw in product_info['keywords'] if kw in summary_lower]
        
        if matching_keywords:
            return f"Recommended because the summary mentions: {', '.join(matching_keywords[:3])}"
        else:
            return f"Recommended based on semantic similarity to {product_info['category']} needs"

def process_test_data(file_path: str):
    """Process test data and generate recommendations for all summaries"""
    # Initialize recommendation system
    rec_system = ProductRecommendationSystem()
    
    # Load test data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
    # Generate recommendations for each summary
    results = []
    
    print("🛍️ Product Recommendation System")
    print("=" * 60)
    
    for i, item in enumerate(data, 1):
        summary = item['summary']
        recommendations = rec_system.get_recommendations(summary, top_k=3)
        
        print(f"\n📋 Summary {i}:")
        print(f"Text: {summary}")
        print(f"\n🎯 Recommended Products:")
        
        summary_result = {
            'id': item['id'],
            'summary': summary,
            'recommendations': []
        }
        
        for j, (product, score, category) in enumerate(recommendations, 1):
            product_name = rec_system.format_product_name(product)
            explanation = rec_system.get_recommendation_explanation(summary, product)
            
            print(f"  {j}. {product_name}")
            print(f"     Category: {category.title()}")
            print(f"     Confidence: {score:.3f}")
            print(f"     Reason: {explanation}")
            
            summary_result['recommendations'].append({
                'rank': j,
                'product': product_name,
                'category': category,
                'confidence_score': round(score, 3),
                'explanation': explanation
            })
        
        results.append(summary_result)
        print("-" * 60)
    
    return results

def save_recommendations_to_file(results: List[Dict], output_file: str):
    """Save recommendations to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    # Process the test data
    test_file = "samsum_data/test.json"
    output_file = "product_recommendations.json"
    
    try:
        results = process_test_data(test_file)
        save_recommendations_to_file(results, output_file)
        
        print(f"\n✅ Successfully processed {len(results)} summaries")
        print(f"📊 Total recommendations generated: {sum(len(r['recommendations']) for r in results)}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {test_file}")
        print("Please make sure the test.json file exists in the samsum_data directory")
    except Exception as e:
        print(f"❌ Error: {str(e)}")