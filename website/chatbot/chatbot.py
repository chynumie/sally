import random
from difflib import get_close_matches
from prettytable import PrettyTable
from datetime import datetime
from groq import Groq
from opcua import Client
import threading
import time
from datetime import datetime
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import socket
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from collections import deque
import warnings

from website.chatbot.intent_handler import get_intent, handle_flow_guidance, handle_oee_improvement, handle_ph_guidance, handle_temperature_guidance
warnings.filterwarnings('ignore')

# GROQ API Key
os.environ["GROQ_API_KEY"] = "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"

# Global variables for models
identify_model = None
identify_scaler = None
when_model = None
when_scaler = None
anomaly_model = None
anomaly_scaler = None

# Define node IDs for OPC UA server
node_ids = {
    "basic": {
        "temperature": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_AI_Scaled.MT_AI2_Temp_Scaled",
        "flowrate": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_AI_Scaled.Comp_Air_Flow_Rate_LPM_SCALED",
        "ph": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_AI_Scaled.MT_AI1_PH_Scaled"
    },
    "oee": {
        "main": {
            "oee": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.OEE",
            "quality": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.OEE_Quality",
            "performance": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.OEE_Performance",
            "availability": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.OEE_Availability"
        },
        "availability": {
            "alarm duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Alarm_Duration",
            "idle duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Idle_Duration",
            "manual duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Manual_Duration",
            "run duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Run_Duration",
            "stop duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Stop_Duration",
            "total duration": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.A_Total_Duration"
        },
        "performance": {
            "average cycle time": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.P_Average_Time",
            "current cycle time": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.P_Current_Time",
            "number of cycles": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.P_Current_Time",
            "previous cycle time": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.P_No_Of_Cycles"
        },
        "quality": {
            "bad bottles": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.Q_No_Of_Bad",
            "good bottles": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.Q_No_Of_Good",
            "total bottles": "ns=4;s=|var|Turck/ARM/WinCE TV.Application.GVL_OEE.Q_Total_No"
        }
    }
}

def is_common_query(text):
    """Check if input contains common conversational phrases"""
    common_phrases = {
        'how are you': ["I'm doing great, thanks for asking! How can I help you today?😊"],
        'what can you do': ["I can help you with:\n1. Checking machine maintenance status\n2. Retrieving live machine data\n3. Answering general questions\nWhat would you like to know?😊"],
        'thank you': ["You're welcome!😊", "Happy to help!😊", "Anytime!😊"],
        'bye': ["Goodbye! Have a great day!😊", "See you later!😊", "Bye! Take care!😊"]
    }
    
    for phrase, responses in common_phrases.items():
        if phrase in text.lower():
            return random.choice(responses)
    return None

def load_models():
    """Load the trained models"""
    global identify_model, identify_scaler, when_model, when_scaler, anomaly_model, anomaly_scaler
    try:
        identify_model = joblib.load('MPidentify_model.pkl')
        identify_scaler = joblib.load('MPidentify_scaler.pkl')
        when_model = joblib.load('MPwhen_model.pkl')
        when_scaler = joblib.load('MPwhen_scaler.pkl')
        anomaly_model = joblib.load('MPanomaly_model.pkl')
        anomaly_scaler = joblib.load('MPanomaly_scaler.pkl')
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        identify_model = None
        identify_scaler = None
        when_model = None
        when_scaler = None
        anomaly_model = None
        anomaly_scaler = None
        return False

def fetch_live_data(query, max_retries=3):
    """Fetch and format current machine data with retry logic"""
    server_url = "opc.tcp://192.168.250.11:4840/"
    client = None
    
    for attempt in range(max_retries):
        try:
            client = Client(server_url)
            client.connection_timeout = 10000
            client.connect()
            
            # Handle basic data queries (pH, temperature, flow rate)
            if query.lower() in node_ids["basic"]:
                node = client.get_node(node_ids["basic"][query.lower()])
                value = node.get_value()
                return {query: f"{value:.2f}"}
            
            # Handle OEE queries
            elif "oee" in query.lower():
                results = {}
                
                # Get main OEE values
                for key, node_id in node_ids["oee"]["main"].items():
                    try:
                        node = client.get_node(node_id)
                        value = node.get_value()
                        results[key] = f"{value}"
                    except Exception as e:
                        print(f"Warning: Failed to fetch {key}: {str(e)}")
                        results[key] = "0"
                
                # Get availability metrics
                for key, node_id in node_ids["oee"]["availability"].items():
                    try:
                        node = client.get_node(node_id)
                        value = node.get_value()
                        results[key] = f"{value}"
                    except Exception as e:
                        print(f"Warning: Failed to fetch {key}: {str(e)}")
                        results[key] = "N/A"
                
                # Get performance metrics
                for key, node_id in node_ids["oee"]["performance"].items():
                    try:
                        node = client.get_node(node_id)
                        value = node.get_value()
                        results[key] = f"{value}"
                    except Exception as e:
                        print(f"Warning: Failed to fetch {key}: {str(e)}")
                        results[key] = "N/A"
                
                # Get quality metrics
                for key, node_id in node_ids["oee"]["quality"].items():
                    try:
                        node = client.get_node(node_id)
                        value = node.get_value()
                        results[key] = f"{value}"
                    except Exception as e:
                        print(f"Warning: Failed to fetch {key}: {str(e)}")
                        results[key] = "N/A"
                
                return results
            
            return "I'm sorry, I couldn't find the requested data."
            
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {"error": f"Failed after {max_retries} attempts: {str(e)}"}
        
        finally:
            if client:
                try:
                    client.disconnect()
                except:
                    pass

def get_current_machine_data(for_prediction=False):
    """Fetch current machine data for maintenance prediction"""
    try:
        # Fetch OEE data
        data = fetch_live_data("oee")
        
        if isinstance(data, dict) and "error" not in data:
            def safe_float(value, default=0.0):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Create DataFrame with all required features
            formatted_data = pd.DataFrame({
                'OEE': safe_float(data.get('oee', 0)),
                'OEE_Quality': safe_float(data.get('quality', 0)),
                'OEE_Performance': safe_float(data.get('performance', 0)),
                'OEE_Availability': safe_float(data.get('availability', 0)),
                'A_Alarm_Duration': safe_float(data.get('alarm duration', 0)),
                'A_Run_Duration': safe_float(data.get('run duration', 0)),
                'A_Stop_Duration': safe_float(data.get('stop duration', 0)),
                'A_Total_Duration': safe_float(data.get('total duration', 0)),
                'P_No_of_Cycles': safe_float(data.get('number of cycles', 0)),
                'Q_No_of_Good': safe_float(data.get('good bottles', 0)),
                'Q_No_of_Bad': safe_float(data.get('bad bottles', 0))
            }, index=[0])
            
            return formatted_data
            
        else:
            print("Error: Invalid or missing OEE data")
            return None
            
    except Exception as e:
        print(f"Error in get_current_machine_data: {str(e)}")
        return None


def get_risk_level(predicted_class, probability=None):
    """Determine risk level based on predicted maintenance timing"""
    if predicted_class == 0:  # Short
        return "HIGH", "🔴"
    elif predicted_class == 1:  # Medium
        return "MEDIUM", "🟠"
    else:  # Long
        return "LOW", "🟢"
    


def handle_maintenance_query(current_data):
    """Handle maintenance predictions using the ML model"""
    try:
        # Scale the data
        X_scaled = when_scaler.transform(current_data)
        
        # Get prediction (0: Short, 1: Medium, 2: Long)
        prediction = when_model.predict(X_scaled)[0]
        probabilities = when_model.predict_proba(X_scaled)[0]
        
        # Get highest probability class and its confidence
        confidence = probabilities[prediction] * 100
        risk_level, emoji = get_risk_level(prediction)
        
        # Convert prediction to time estimate
        time_estimates = {
            0: "< 1 week",    # Short
            1: "1-2 weeks",   # Medium
            2: "> 2 weeks"    # Long
        }
        
        response = f"Maintenance Status: {emoji}\n"
        response += f"Risk Level: {risk_level}\n"
        response += f"Confidence: {confidence:.1f}%\n"
        response += f"Estimated Time to Maintenance: {time_estimates[prediction]}\n"
        
        # Add recommendations based on prediction
        if prediction == 0:  # Short
            response += "\nRecommendations:\n"
            response += "• Schedule maintenance within this week\n"
            response += "• Monitor system closely\n"
            response += "• Prepare maintenance resources\n"
        elif prediction == 1:  # Medium
            response += "\nRecommendations:\n"
            response += "• Plan maintenance within next 2 weeks\n"
            response += "• Monitor performance metrics\n"
            response += "• Begin maintenance preparations\n"
        else:  # Long
            response += "\nRecommendations:\n"
            response += "• Continue regular monitoring\n"
            response += "• Follow standard maintenance schedule\n"
            response += "• Document any performance changes\n"
        
        # Add confidence-based warning if needed
        if confidence < 70:
            response += "\n⚠️ Note: Prediction confidence is lower than usual.\n"
            response += "Consider additional monitoring and data collection.\n"
            
        return response
        
    except Exception as e:
        print(f"Error in maintenance prediction: {str(e)}")
        return "Error making maintenance prediction"
    

# Add new function for anomaly detection
def handle_anomaly_detection(current_data):
    """Handle anomaly detection using the ML model"""
    try:
        # Scale the data
        X_scaled = anomaly_scaler.transform(current_data)
        
        # Get prediction (0: Abnormal, 1: Normal)
        prediction = anomaly_model.predict(X_scaled)[0]
        probabilities = anomaly_model.predict_proba(X_scaled)[0]
        
        # Get confidence
        confidence = probabilities[prediction] * 100
        
        # Prepare response
        status = "Normal" if prediction == 1 else "Abnormal"
        emoji = "✅" if prediction == 1 else "⚠️"
        
        response = f"System Status: {emoji}\n"
        response += f"Current State: {status}\n"
        response += f"Confidence: {confidence:.1f}%\n"
        
        # Add recommendations based on status
        if prediction == 0:  # Abnormal
            response += "\nRecommendations:\n"
            response += "• Check system parameters\n"
            response += "• Review recent changes\n"
            response += "• Monitor closely\n"
        
        # Add confidence-based warning if needed
        if confidence < 70:
            response += "\n⚠️ Note: Prediction confidence is lower than usual.\n"
            response += "Consider additional monitoring.\n"
            
        return response
        
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return "Error performing anomaly detection"



def display_maintenance_prediction(prediction_response):
    """Format and display maintenance prediction"""
    table = PrettyTable()
    table.title = "Maintenance Prediction"
    table.field_names = ["Metric", "Value"]
    
    # Parse the response string into lines
    lines = prediction_response.split('\n')
    
    for line in lines:
        if line.strip():  # Skip empty lines
            if ':' in line:
                key, value = line.split(':', 1)
                table.add_row([key.strip(), value.strip()])
            elif '•' in line:
                table.add_row(["Recommendation", line.strip('• ')])
            else:
                table.add_row(['Note', line.strip()])
    
    print("\n")
    print(table)

def display_oee_table(oee_data):
    """Display OEE data in formatted tables"""
    # Create tables
    main_table = PrettyTable()
    availability_table = PrettyTable()
    performance_table = PrettyTable()
    quality_table = PrettyTable()

    # Main OEE values
    main_table.title = "OEE Overview"
    main_table.field_names = ["Metric", "Value"]
    main_components = ["oee", "availability", "performance", "quality"]
    for component in main_components:
        main_table.add_row([component.upper(), oee_data.get(component, "N/A")])

    # Availability metrics
    availability_table.title = "Availability Metrics"
    availability_table.field_names = ["Metric", "Value"]
    availability_components = ["alarm duration", "run duration", "stop duration", "total duration"]
    for component in availability_components:
        availability_table.add_row([component.title(), oee_data.get(component, "N/A")])

    # Performance metrics
    performance_table.title = "Performance Metrics"
    performance_table.field_names = ["Metric", "Value"]
    performance_components = ["number of cycles"]
    for component in performance_components:
        performance_table.add_row([component.title(), oee_data.get(component, "N/A")])

    # Quality metrics
    quality_table.title = "Quality Metrics"
    quality_table.field_names = ["Metric", "Value"]
    quality_components = ["good bottles", "bad bottles"]
    for component in quality_components:
        quality_table.add_row([component.title(), oee_data.get(component, "N/A")])

    # Print all tables with spacing
    print("\n")
    print(main_table)
    print("\n")
    print(availability_table)
    print("\n")
    print(performance_table)
    print("\n")
    print(quality_table)

def handle_oee_query(user_input, oee_data):
    """Handle OEE queries and display data"""
    user_input = user_input.lower()
    
    # Always display the full OEE table for 'oee' queries
    display_oee_table(oee_data)
    
    # If specific component mentioned, also show that value
    for component in oee_data.keys():
        if component.lower() in user_input and component.lower() != 'oee':
            return f"\nSpecific component: {component.capitalize()}: {oee_data[component]}"
    
    return None

def check_connection(max_attempts=3):
    """Check connection to machine with limited attempts"""
    for attempt in range(max_attempts):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            print("Internet connection established")
            return True
        except OSError:
            if attempt < max_attempts - 1:
                print(f"Connection attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
            else:
                print("Connection failed after maximum attempts. Operating in offline mode.")
    return False

def spell_correct(user_input, valid_keywords):
    """Correct spelling in user input"""
    exclusions = {"pH", "OEE"}
    corrected_input = user_input.lower().strip()
    
    closest_match = None
    for keyword in valid_keywords:
        if corrected_input in keyword.lower():
            closest_match = keyword
            break
    
    return closest_match if closest_match else corrected_input


# InfluxDB Configuration

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class OEECloudManager:
    def __init__(self):
        """Initialize InfluxDB client only if internet is available"""
        self.client = None
        self.write_api = None
        self.query_api = None
        
        # Only initialize if internet is available
        if check_connection():
            try:
                # InfluxDB configuration
                self.bucket = "_monitoring"
                self.org = "Temasek Polytechnic"
                self.token = "bZZUQJ2rX7AuhpASbKIoxXrVMzFZ9aRijklrtlfrIu6bYAKiJ7rAIwe14XW0JeZ6mc_js387HvzO-evKcsZHNQ=="
                self.url = "https://us-east-1-1.aws.cloud2.influxdata.com/"
                
                # Initialize client
                self.client = influxdb_client.InfluxDBClient(
                    url=self.url,
                    token=self.token,
                    org=self.org
                )
                
                # Initialize APIs
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                self.query_api = self.client.query_api()
                print("InfluxDB connection established")
            except Exception as e:
                print(f"Warning: InfluxDB initialization failed: {e}")
        else:
            print("No internet connection - InfluxDB features disabled")

    def write_oee_data(self, data_dict, timestamp=None):
        """Write OEE metrics to InfluxDB if available"""
        if not self.write_api:
            return False
            
        if timestamp is None:
            timestamp = datetime.utcnow()

        try:
            point = influxdb_client.Point("oee_metrics")\
                .tag("machine_id", "machine_1")\
                .tag("location", "production_line_1")
                
            for key, value in data_dict.items():
                try:
                    point = point.field(key, float(value))
                except (ValueError, TypeError):
                    point = point.field(key, str(value))
                    
            point = point.time(timestamp)
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
        except Exception as e:
            print(f"Warning: Failed to write to InfluxDB: {e}")
            return False

    def get_historical_data(self, hours=168):  # Default to 1 week
        """Retrieve historical OEE data if available"""
        if not self.query_api:
            return None
            
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -{hours}h)
                |> filter(fn: (r) => r["_measurement"] == "oee_metrics")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = self.query_api.query_data_frame(query)
            if result.empty:
                return None
                
            # Clean and process the DataFrame
            result = result.drop(columns=['result', 'table', '_measurement', '_start', '_stop'])
            result = result.rename(columns={'_time': 'timestamp'})
            return result
                
        except Exception as e:
            print(f"Warning: Failed to query InfluxDB: {e}")
            return None

    def get_trend_analysis(self, days=7):
        """Analyze OEE trends if data is available"""
        if not self.query_api:
            return None
            
        df = self.get_historical_data(hours=days*24)
        if df is None or df.empty:
            return None
            
        try:
            analysis = {
                'oee': {
                    'current': df['oee'].iloc[-1],
                    'mean': df['oee'].mean(),
                    'trend': df['oee'].iloc[-1] - df['oee'].iloc[0],
                    'volatility': df['oee'].std()
                },
                'availability': {
                    'current': df['availability'].iloc[-1],
                    'mean': df['availability'].mean(),
                    'trend': df['availability'].iloc[-1] - df['availability'].iloc[0]
                },
                'performance': {
                    'current': df['performance'].iloc[-1],
                    'mean': df['performance'].mean(),
                    'trend': df['performance'].iloc[-1] - df['performance'].iloc[0]
                },
                'quality': {
                    'current': df['quality'].iloc[-1],
                    'mean': df['quality'].mean(),
                    'trend': df['quality'].iloc[-1] - df['quality'].iloc[0]
                }
            }
            return analysis
        except Exception as e:
            print(f"Warning: Failed to analyze trends: {e}")
            return None

    def get_maintenance_features(self):
        """Get maintenance prediction features if data is available"""
        if not self.query_api:
            return None
            
        df = self.get_historical_data(hours=168)  # Get 1 week of data
        if df is None or df.empty:
            return None
            
        try:
            features = {
                'OEE_4hour_max': df['oee'].rolling(window=4).max().iloc[-1],
                'OEE_4hour_min': df['oee'].rolling(window=4).min().iloc[-1],
                'OEE_volatility': df['oee'].std(),
                'A_Alarm_Duration': df['alarm_duration'].sum(),
                'A_Run_Duration': df['run_duration'].sum(),
                'A_Stop_Duration': df['stop_duration'].sum(),
                'Alarm_Rate': df['alarm_duration'].sum() / df['total_duration'].sum(),
                'Cycle_Efficiency': df['performance'].mean(),
                'Downtime_Ratio': df['stop_duration'].sum() / df['total_duration'].sum(),
                'Quality_Rate': df['quality'].mean()
            }
            return pd.DataFrame([features])
        except Exception as e:
            print(f"Warning: Failed to generate maintenance features: {e}")
            return None

def update_cloud_data(oee_data):
    """Update cloud data with current OEE metrics"""
    try:
        cloud_manager = OEECloudManager()
        
        # Prepare data dictionary
        data_dict = {
            'oee': oee_data.get('oee', 0),
            'availability': oee_data.get('availability', 0),
            'performance': oee_data.get('performance', 0),
            'quality': oee_data.get('quality', 0),
            'alarm_duration': oee_data.get('alarm duration', 0),
            'run_duration': oee_data.get('run duration', 0),
            'stop_duration': oee_data.get('stop duration', 0),
            'total_duration': oee_data.get('total duration', 0),
            'cycles': oee_data.get('number of cycles', 0),
            'good_bottles': oee_data.get('good bottles', 0),
            'bad_bottles': oee_data.get('bad bottles', 0)
        }
        
        return cloud_manager.write_oee_data(data_dict)
    except Exception as e:
        print(f"Error updating cloud data: {e}")
        return False

# Example usage in your main code:
def get_maintenance_prediction_with_history():
    """
    Get maintenance prediction using both live and historical data
    """
    cloud_manager = OEECloudManager()
    
    # Get current machine data
    current_data = get_current_machine_data(for_prediction=True)
    
    # Get historical features
    historical_features = cloud_manager.get_maintenance_features()
    
    if current_data is not None and historical_features is not None:
        # Combine current and historical features
        combined_features = pd.concat([current_data, historical_features], axis=1)
        
        # Make prediction using combined features
        if when_model is not None and when_scaler is not None:
            scaled_features = when_scaler.transform(combined_features)
            prediction = when_model.predict(scaled_features)[0]
            probabilities = when_model.predict_proba(scaled_features)[0]
            
            # Get trend data
            trend_data = cloud_manager.get_trend_analysis(days=7)
            
            return prediction, probabilities, trend_data
            
    return None, None, None


def start_data_collection():
    """Start periodic data collection"""
    def collect_data():
        while True:
            try:
                live_data = fetch_live_data("oee")
                if isinstance(live_data, dict):
                    update_cloud_data(live_data)
                time.sleep(300)  # Update every 5 minutes
            except Exception as e:
                print(f"Error in data collection: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    # Start data collection in a separate thread
    collection_thread = threading.Thread(target=collect_data, daemon=True)
    collection_thread.start()
    return collection_thread

def display_historical_data(data):
    """Display historical data analysis in a formatted table"""
    table = PrettyTable()
    table.field_names = ["Metric", "Current", "Mean", "Trend"]
    
    for metric, values in data.items():
        table.add_row([
            metric.upper(),
            f"{values.get('current', 'N/A'):.2f}",
            f"{values.get('mean', 'N/A'):.2f}",
            f"{values.get('trend', 'N/A'):.2f}"
        ])
    
    print(table)

def cleanup():
    """Cleanup function that doesn't assume client exists"""
    try:
        # Only try to close if client exists and has close method
        if hasattr(cleanup, 'client') and hasattr(cleanup.client, 'close'):
            cleanup.client.close()
            print("\nClosed connections and cleaned up resources.")
    except Exception as e:
        print(f"\nError during cleanup: {e}")

# Initialize client at module level
try:
    client = Client("opc.tcp://192.168.250.11:4840/")
    cleanup.client = client  # Store client reference for cleanup
except Exception as e:
    print(f"Warning: Failed to initialize OPC UA client: {e}")
    client = None




def explain_oee_results(oee_data):
    """Explain OEE results in detail"""
    explanation = "\nHere's what these numbers mean:\n"
    
    if 'oee' in oee_data:
        oee = float(oee_data['oee'])
        if oee >= 85:
            explanation += "• Your OEE is excellent! World-class OEE is considered to be 85% or higher.\n"
        elif oee >= 60:
            explanation += "• Your OEE is average. There's room for improvement to reach world-class standards.\n"
        else:
            explanation += "• Your OEE is below average. This suggests significant improvement opportunities.\n"
    
    if 'availability' in oee_data:
        avail = float(oee_data['availability'])
        if avail < 90:
            explanation += "• Low availability suggests excessive downtime. Check for frequent stops or breakdowns.\n"
    
    if 'performance' in oee_data:
        perf = float(oee_data['performance'])
        if perf < 95:
            explanation += "• Performance loss might be due to slow cycles or minor stops.\n"
    
    if 'quality' in oee_data:
        qual = float(oee_data['quality'])
        if qual < 99:
            explanation += "• Quality losses indicate defects or rework requirements.\n"
    
    return explanation

def explain_parameters(current_data):
    """Explain current parameter readings"""
    explanation = "\nHere's what these readings indicate:\n"
    
    # pH explanation
    if 'MT_AI1_PH_Scaled' in current_data:
        ph = current_data['MT_AI1_PH_Scaled']
        if 5.5 <= ph <= 6.5:
            explanation += "• pH is within optimal range (5.5-6.5)\n"
        elif ph < 5.5:
            explanation += "• pH is too acidic. This might cause corrosion issues.\n"
        else:
            explanation += "• pH is too basic. This could lead to scaling problems.\n"
    
    # Temperature explanation
    if 'MT_AI2_Temp_Scaled' in current_data:
        temp = current_data['MT_AI2_Temp_Scaled']
        if 28 <= temp <= 32:
            explanation += "• Temperature is optimal (28-32°C)\n"
        elif temp < 28:
            explanation += "• Temperature is low. This might reduce reaction efficiency.\n"
        else:
            explanation += "• Temperature is high. This could affect product quality.\n"
    
    # Flow rate explanation
    if 'Flowmeter_Totalizer' in current_data:
        flow = current_data['Flowmeter_Totalizer']
        if 98 <= flow <= 102:
            explanation += "• Flow rate is optimal (98-102)\n"
        elif flow < 98:
            explanation += "• Low flow rate might reduce production efficiency.\n"
        else:
            explanation += "• High flow rate could cause turbulence issues.\n"
    
    return explanation

def explain_maintenance_prediction(prediction, confidence):
    """Explain maintenance prediction"""
    explanation = "\nHere's what this prediction means:\n"
    
    if prediction == 0:  # Short
        explanation += "• Urgent maintenance needed due to:\n"
        explanation += "  - Possible equipment wear patterns\n"
        explanation += "  - Performance degradation indicators\n"
        explanation += "  - Historical maintenance patterns\n"
    elif prediction == 1:  # Medium
        explanation += "• Moderate maintenance timeline due to:\n"
        explanation += "  - Early warning signs present\n"
        explanation += "  - Some performance metrics declining\n"
        explanation += "  - Preventive maintenance timing\n"
    else:  # Long
        explanation += "• Equipment is running well because:\n"
        explanation += "  - All parameters within normal ranges\n"
        explanation += "  - No significant wear indicators\n"
        explanation += "  - Good historical performance\n"
    
    if confidence < 70:
        explanation += "\nNote: The lower confidence level suggests monitoring additional parameters.\n"
    
    return explanation



def process_message(user_message):
    """Process a single message and return response"""
    # Load models if not already loaded
    global identify_model, identify_scaler, when_model, when_scaler
    if identify_model is None:
        load_models()
    
    user_message = user_message.lower()
    
    # Check for maintenance-related queries
    maintenance_keywords = ['maintenance', 'repair', 'fix', 'service', 'condition', 'health']
    if any(keyword in user_message for keyword in maintenance_keywords):
        try:
            # Get current machine data
            current_data = get_current_machine_data(for_prediction=True)
            
            if current_data is not None:
                # Make prediction
                maintenance_response = handle_maintenance_query(current_data)
                return maintenance_response
            else:
                return "I'm having trouble getting the current machine data. Please check the machine connection and try again. 😊"
                
        except Exception as e:
            print(f"Error in maintenance prediction: {str(e)}")
            return "I encountered an error while checking maintenance status. Please try again later. 😊"
    
    # Check for common queries
    common_response = is_common_query(user_message)
    if common_response:
        return common_response
    
    # If no specific response, return None to let views.py handle it with Groq
    return "I can help you with that! 😊"










def chatbot():
    """Main chatbot function"""
    
    print("Chatbot: Hello, I am Sally! How can I help you today!😊")

    global identify_model, identify_scaler, when_model, when_scaler
    
    # Load maintenance models at startup
    if not load_models():
        print("Warning: Models failed to load. Some features may be unavailable.")
    
    # Check connection once at startup with limited attempts
    is_connected = False
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            is_connected = True
            print("Connection to machine established")
            break
        except OSError:
            if attempt < max_attempts - 1:
                print(f"Connection attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
            else:
                print("Connection failed after maximum attempts. Operating in offline mode.")
    
    # Initialize services based on connection status
    cloud_manager = None
    groq_client = None
    
    if is_connected:
        try:
            cloud_manager = OEECloudManager()
            groq_client = Groq()
            print("Online services initialized successfully")
        except Exception as e:
            print(f"Warning: Some online features unavailable: {e}")
    else:
        print("Operating in offline mode - cloud features and AI responses unavailable")

    try:
        while True:        
            user_input = input("\nYou: ").strip().lower()
            
            # Check for quit commands
            quit_keywords = ['exit', 'quit', 'bye', 'goodbye', 'x']
            corrected_query = spell_correct(user_input, quit_keywords)
            if corrected_query.lower() in quit_keywords:
                goodbye_responses = [
                    "\nChatbot: Goodbye! Thank you for chatting with me. Have a great day! 😊",
                    "\nChatbot: Thanks for the conversation! Feel free to come back if you need more help. Goodbye! 😊",
                    "\nChatbot: It was great helping you today! Take care and goodbye! 😊",
                    "\nChatbot: Farewell! Don't hesitate to return if you need more assistance with your manufacturing process! 😊"
                ]
                print(random.choice(goodbye_responses))
                break 
            
            # Check for greetings first
            greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
            greeting_responses = [
                "Hi there! How can I help you today?😊",
                "Hello! What can I do for you?😊",
                "Hey! How may I assist you?😊",
                "Hi! I'm here to help!😊"
            ]
            if any(greeting in user_input for greeting in greeting_keywords):
                print(f"\nChatbot: {random.choice(greeting_responses)}")
                continue

            # Get the intent
            intent = get_intent(user_input)
            
            # Handle based on intent
            if intent == 'oee_improvement':
                response = handle_oee_improvement()
                print(f"\nChatbot: {response}")
                
                # Try to get live data if available
                if is_connected:
                    try:
                        print("\nLet me check the current OEE data...")
                        live_data = fetch_live_data("oee")
                        if isinstance(live_data, dict):
                            handle_oee_query(user_input, live_data)
                    except Exception as e:
                        print(f"\nNote: Could not fetch live data ({str(e)})")
                
            elif intent == 'ph_guidance':
                response = handle_ph_guidance()
                print(f"\nChatbot: {response}")
                
                # Try to get live pH data if available
                if is_connected:
                    try:
                        print("\nLet me check the current pH readings...")
                        current_data = get_current_machine_data()
                        if current_data is not None:
                            print(f"Current pH: {current_data['MT_AI1_PH_Scaled']:.2f}")
                    except Exception as e:
                        print(f"\nNote: Could not fetch live data ({str(e)})")
                
            elif intent == 'temperature_guidance':
                response = handle_temperature_guidance()
                print(f"\nChatbot: {response}")
                
                # Try to get live temperature data if available
                if is_connected:
                    try:
                        print("\nLet me check the current temperature...")
                        current_data = get_current_machine_data()
                        if current_data is not None:
                            print(f"Current Temperature: {current_data['MT_AI3_Temp_Scaled']:.2f}°C")
                    except Exception as e:
                        print(f"\nNote: Could not fetch live data ({str(e)})")

            elif intent == 'flowrate_guidance':
                response = handle_flow_guidance()
                print(f"\nChatbot: {response}")
                
                # Try to get live flowrate data if available
                if is_connected:
                    try:
                        print("\nLet me check the current temperature...")
                        current_data = get_current_machine_data()
                        if current_data is not None:
                            print(f"Current Flowrate: {current_data['Flowmeter_Totalizer']:.2f}°C")
                    except Exception as e:
                        print(f"\nNote: Could not fetch live data ({str(e)})")
            
            # Handle based on connection status first
            if not is_connected and any(word in user_input for word in ['data', 'current', 'now', 'status']):
                print("\nChatbot: Sorry, I cannot access machine data in offline mode.")
                print("I can still help you with:")
                print("1. General OEE information")
                print("2. Maintenance guidelines")
                print("3. Best practices")
                print("\nWhat would you like to know about?")
                continue

            if any(word in user_input.lower() for word in ['anomaly', 'anomalies', 'abnormal', 'status', 'state', 'wrong']):
                current_data = get_current_machine_data()
                if current_data is not None:
                    print("\nWould you like me to explain what these readings indicate?")
                    if input("You: ").lower().strip() in ['yes', 'y', 'yes please']:
                        explanation = explain_parameters(current_data)
                        print(f"Chatbot: {explanation}")

            # Check for maintenance queries
            if "maintenance" in user_input:
                if not is_connected:
                    print("\nChatbot: I cannot check maintenance status without connection.")
                    print("Would you like to learn about maintenance best practices instead?")
                    continue
                    
                print("\nChatbot: Analyzing maintenance requirements...")
                prediction, probabilities, trend_data = get_maintenance_prediction_with_history()
                
                if prediction is not None:
                    maintenance_response = handle_maintenance_query(prediction, probabilities)
                    display_maintenance_prediction(maintenance_response)
                    print("\nWould you like me to explain this prediction in detail?")
                    if input("You: ").lower().strip() in ['yes', 'y', 'yes please']:
                        explanation = explain_maintenance_prediction(prediction, probabilities[prediction] * 100)
                        print(f"Chatbot: {explanation}")
                    
                    # Show historical trend if available
                    if trend_data:
                        print("\nTrend Analysis:")
                        for metric, values in trend_data.items():
                            print(f"\n{metric.upper()}:")
                            print(f"Current: {values['current']:.2f}")
                            print(f"7-day trend: {values['trend']:.2f}")
                            print(f"Volatility: {values['volatility']:.2f}")
                else:
                    print("\nChatbot: Unable to perform maintenance prediction at this time.")
                continue

            # Check for live data queries
            if any(keyword in user_input for keyword in list(node_ids["basic"].keys()) + ["oee"]):
                if not is_connected:
                    print("\nChatbot: Cannot fetch live data without connection.")
                    continue
                    
                corrected_query1 = spell_correct(user_input, list(node_ids["basic"].keys()) + ["oee"])
                live_data = fetch_live_data(corrected_query1)
                
                if isinstance(live_data, dict):
                    if "error" in live_data:                
                        print(f"\nChatbot: Unable to retrieve data. {live_data['error']}")
                    elif "oee" in corrected_query1.lower():
                        specific_response = handle_oee_query(user_input, live_data)
                        if specific_response:
                            print(f"\nChatbot: {specific_response}")
                            print("\nWould you like me to explain what these numbers mean?")
                            if input("You: ").lower().strip() in ['yes', 'y', 'yes please']:
                                explanation = explain_oee_results(live_data)
                                print(f"Chatbot: {explanation}")
                            
                            # Show historical comparison if available
                            if cloud_manager:
                                historical_data = cloud_manager.get_trend_analysis(days=1)
                                if historical_data and 'oee' in historical_data:
                                    print("\nLast 24 hours comparison:")
                                    print(f"Average OEE: {historical_data['oee']['mean']:.2f}%")
                                    print(f"Trend: {'+' if historical_data['oee']['trend'] > 0 else ''}{historical_data['oee']['trend']:.2f}%")
                    else:
                        response = "Here's the current data:\n" + "\n".join(
                            f"• {key.replace('_', ' ').title()}: {value}" 
                            for key, value in live_data.items()
                        )
                        print(f"\nChatbot: {response}")
                else:
                    print(f"\nChatbot: {live_data}")
                continue

            # Check for historical data queries
            if any(word in user_input for word in ['history', 'historical', 'trend', 'past']):
                if not is_connected or not cloud_manager:
                    print("\nChatbot: Historical data is not available without connection.")
                    continue
                    
                try:
                    if 'week' in user_input or '7 days' in user_input:
                        days = 7
                    elif 'month' in user_input or '30 days' in user_input:
                        days = 30
                    else:
                        days = 1
                        
                    historical_data = cloud_manager.get_trend_analysis(days=days)
                    if historical_data:
                        print(f"\nHistorical Data Analysis (Last {days} {'days' if days > 1 else 'day'}):")
                        display_historical_data(historical_data)
                    else:
                        print("\nChatbot: No historical data available for the requested period.")
                except Exception as e:
                    print(f"\nChatbot: Error retrieving historical data: {str(e)}")
                continue

            # If no specific keywords matched, use Groq for general conversation
            if is_connected and groq_client:
                try:
                    messages = [
                        {"role": "system", "content": """You are Sally, a helpful and friendly chatbot assistant 
                        specializing in manufacturing and OEE optimization. Your responses should be:
                        1. Focused on manufacturing/OEE context
                        2. Practical and actionable
                        3. Clear and concise
                        4. Professional yet friendly and helpful"""},
                        {"role": "user", "content": user_input}
                    ]
                    
                    completion = groq_client.chat.completions.create(
                        model="mixtral-8x7b-32768",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=512,
                        top_p=1,
                        stream=True
                    )
                    
                    print("\nChatbot:", end=" ", flush=True)
                    for chunk in completion:
                        if hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                            if content:
                                print(content, end="", flush=True)
                    print()  # New line after response
                    
                except Exception as e:
                    print(f"\nChatbot: AI response error: {str(e)}")
                    print("Would you like to ask about specific machine metrics instead?")
            else:
                print("\nChatbot: I can help you with:")
                print("1. General OEE information")
                print("2. Maintenance guidelines")
                print("3. Best practices")
                print("\nWhat would you like to know more about?")
                
    except KeyboardInterrupt:
        print("\nShutting down chatbot...")
    finally:
        cleanup()

if __name__ == "__main__":
    chatbot()

