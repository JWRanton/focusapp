from flask import Flask, render_template, request
import logging
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the text generation pipeline with a suitable model
generator = pipeline('text-generation', model='gpt2')

def generate_breakdown(task):
    logger.info(f"Generating breakdown for task: {task}")
    
    prompt = (
        f"Break down the task '{task}' into 5 clear, actionable steps:\n"
    )
    
    try:
        # Generate the breakdown
        result = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7)
        generated_text = result[0]['generated_text']
        
        # Process the generated text
        steps = generated_text.split('\n')[1:]  # Skip the prompt
        formatted_steps = []
        for step in steps:
            cleaned_step = step.strip()
            if cleaned_step:
                formatted_steps.append(f"• {cleaned_step}")
        
        breakdown = "\n".join(formatted_steps)
        logger.info("Successfully generated breakdown")
        return breakdown
    except Exception as e:
        logger.error(f"Error generating breakdown: {e}")
        return f"Error: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        task = request.form['task']
        logger.info(f"Received task: {task}")
        breakdown = generate_breakdown(task)
        return render_template('index.html', task=task, breakdown=breakdown)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
