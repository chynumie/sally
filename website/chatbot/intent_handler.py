def get_intent(user_input):
    """Determine the user's intent from their input"""
    # Define intent patterns
    intents = {
        'oee_improvement': {
            'keywords': ['improve', 'increase', 'better', 'optimize', 'enhance', 'efficiency'],
            'required': ['oee', 'efficiency', 'performance']
        },
        'ph_guidance': {
            'keywords': ['ph', 'acid', 'base', 'acidic', 'basic'],
            'required': []
        },
        'temperature_guidance': {
            'keywords': ['temperature', 'temp', 'heat', 'cooling', 'thermal'],
            'required': []
        },
        'maintenance_check': {
            'keywords': ['maintenance', 'repair', 'fix', 'service', 'check'],
            'required': []
        },
        'flow_guidance': { 
            'keywords': ['flow', 'flowrate', 'flowmeter', 'rate', 'flow rate'],
            'required': []
        }
    }
    
    # Check each intent
    for intent, pattern in intents.items():
        if any(keyword in user_input.lower() for keyword in pattern['keywords']):
            if not pattern['required'] or any(req in user_input.lower() for req in pattern['required']):
                return intent
    
    return 'general_query'

def handle_oee_improvement():
    """Handle OEE improvement queries"""
    response = """Here are specific ways to improve OEE:

1. Availability Improvement:
   • Reduce unplanned downtime
   • Optimize maintenance schedules
   • Implement quick changeover procedures
   • Monitor and address frequent stoppages

2. Performance Improvement:
   • Optimize cycle times
   • Reduce minor stops
   • Address speed losses
   • Maintain consistent production rates

3. Quality Improvement:
   • Reduce defects and rework
   • Implement quality checks
   • Enhance operator training
   • Monitor and control process parameters

Would you like specific details about any of these areas?"""
    return response

def handle_ph_guidance():
    """Handle pH-related queries"""
    response = """Here's guidance on pH management:

1. Optimal Range:
   • Maintain pH between 5.5 and 6.5
   • Monitor regularly
   • Implement automatic pH control when possible

2. Common Issues:
   • Sudden pH changes may indicate process problems
   • High pH can cause scaling
   • Low pH can cause corrosion

3. Best Practices:
   • Regular calibration of pH sensors
   • Proper maintenance of dosing systems
   • Staff training on pH control
   • Documentation of pH trends

Need more specific information about pH control?"""
    return response

def handle_temperature_guidance():
    """Handle temperature-related queries"""
    response = """Temperature Management Guidelines:

1. Optimal Range:
   • Maintain temperature between 28-32°C
   • Monitor for sudden changes
   • Regular calibration of temperature sensors

2. Impact on Process:
   • Too high: Can affect product quality
   • Too low: May reduce efficiency
   • Fluctuations: Can indicate system issues

3. Best Practices:
   • Regular monitoring
   • Preventive maintenance of cooling systems
   • proper insulation
   • Temperature trend analysis

Would you like more details about temperature control?"""
    return response


def handle_flow_guidance():
    """Handle flow-related queries"""
    response = """Flow Rate Management Guidelines:

1. Optimal Range:
   • Maintain flow rate between 98-102 units
   • Monitor for sudden fluctuations
   • Regular calibration of flowmeters

2. Impact on Process:
   • Too high: Can cause turbulence and quality issues
   • Too low: May reduce production efficiency
   • Inconsistent flow: Can affect product quality

3. Best Practices:
   • Regular monitoring of flow rates
   • Preventive maintenance of pumps and valves
   • Check for pipe blockages or restrictions
   • Monitor pressure drops

4. Common Issues:
   • Pump cavitation
   • Valve wear
   • Sensor calibration drift
   • Pipeline restrictions

Would you like more specific information about flow control?"""
    return response
    
