from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField, HiddenField
from wtforms.validators import DataRequired


class AddDecisionForm(FlaskForm):
    """Form for adding a new decision."""
    problem = TextAreaField('Problem or Situation', validators=[DataRequired()])
    options = TextAreaField('Options (one per line)', validators=[DataRequired()])
    chosen = StringField('Your Choice', validators=[DataRequired()])
    reasoning = TextAreaField('Your Reasoning (optional)')
    mood = StringField('Your Mood/Emotional State (optional)')
    submit = SubmitField('Add Decision')


class PredictDecisionForm(FlaskForm):
    """Form for predicting a decision."""
    problem = TextAreaField('Problem or Situation', validators=[DataRequired()])
    options = TextAreaField('Options (one per line)', validators=[DataRequired()])
    submit = SubmitField('Predict My Decision')


class FeedbackForm(FlaskForm):
    """Form for providing feedback on a prediction."""
    problem = HiddenField('Problem')
    options = HiddenField('Options')
    prediction = HiddenField('Prediction')
    correct = SelectField('Was this prediction correct?', 
                         choices=[('yes', 'Yes'), ('no', 'No')])
    actual_choice = StringField('What would be your actual choice?')
    actual_reasoning = TextAreaField('What would be your actual reasoning?')
    actual_mood = StringField('What was your mood/emotional state? (optional)')
    submit = SubmitField('Submit Feedback')