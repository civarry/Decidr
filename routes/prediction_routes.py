import json
from flask import Blueprint, render_template, request, redirect, url_for, flash

from routes.decision_routes import decision_mirror
from forms.forms import PredictDecisionForm, FeedbackForm

# Create blueprint
prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict a decision."""
    form = PredictDecisionForm()
    prediction = None

    if form.validate_on_submit():
        options = [opt.strip() for opt in form.options.data.strip().split('\n') if opt.strip()]

        try:
            # Run prediction
            raw_prediction = decision_mirror.predict_decision(
                problem=form.problem.data,
                options=options
            )

            # Sanitize or parse response (assume JSON format)
            if isinstance(raw_prediction, str):
                try:
                    # Attempt to parse stringified JSON
                    prediction = json.loads(raw_prediction)
                except json.JSONDecodeError:
                    # If it's not JSON, pass as raw string
                    prediction = {'raw': raw_prediction}
            elif isinstance(raw_prediction, dict):
                prediction = raw_prediction
            else:
                prediction = {'raw': str(raw_prediction)}

        except Exception as e:
            flash(f"Error during prediction: {e}", 'danger')
            return redirect(url_for('prediction.predict'))

        return render_template('prediction_result.html',
                              prediction=prediction,
                              problem=form.problem.data,
                              options=options)

    return render_template('predict.html', form=form)

@prediction_bp.route('/feedback', methods=['POST'])
def feedback():
    """Process feedback on a prediction."""
    form = FeedbackForm()
    
    if form.validate_on_submit():
        try:
            # Process options from hidden field
            if form.options.data and form.options.data.strip():
                # Split the options by newline
                options = [opt.strip() for opt in form.options.data.strip().split('\n') if opt.strip()]
            else:
                options = []  # Handle empty options field
            
            if form.correct.data == 'no':
                # Add the correct decision
                decision = decision_mirror.add_decision(
                    problem=form.problem.data,
                    options=options,
                    chosen=form.actual_choice.data,
                    reasoning=form.actual_reasoning.data or None,
                    mood=form.actual_mood.data or None
                )
                flash('Thanks for your feedback! A new decision has been added based on your input.', 'success')
            else:
                flash('Thanks for confirming the prediction was correct!', 'success')
            
            return redirect(url_for('decision.view_decisions'))
        except Exception as e:
            # Catch-all for any other errors
            flash(f'An error occurred: {str(e)}', 'danger')
            print(f"Error in feedback route: {str(e)}")
            return redirect(url_for('prediction.predict'))
    
    return redirect(url_for('prediction.predict'))