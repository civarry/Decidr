from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from typing import List

from config import Config
from core.decision_mirror import DecisionMirror
from forms.forms import AddDecisionForm

# Create blueprint
decision_bp = Blueprint('decision', __name__)

# Initialize DecisionMirror singleton
decision_mirror = DecisionMirror(
    api_key=Config.GROQ_API_KEY,
    model=Config.LLM_MODEL,
    data_dir=Config.DATA_DIR
)

@decision_bp.route('/')
def index():
    """Homepage with dashboard."""
    # Get counts for dashboard
    decision_count = len(decision_mirror.decisions)
    # Get recent decisions for quick access
    recent_decisions = decision_mirror.get_decisions(limit=3)
    return render_template(
        'index.html', 
        decision_count=decision_count,
        recent_decisions=recent_decisions
    )

@decision_bp.route('/add_decision', methods=['GET', 'POST'])
def add_decision():
    """Add a new decision."""
    form = AddDecisionForm()
    if form.validate_on_submit():
        # Process options from textarea (one per line)
        options = [opt.strip() for opt in form.options.data.strip().split('\n') if opt.strip()]
        
        # Add decision
        decision = decision_mirror.add_decision(
            problem=form.problem.data,
            options=options,
            chosen=form.chosen.data,
            reasoning=form.reasoning.data or None,
            mood=form.mood.data or None
        )
        
        flash(f'Decision #{decision.id} added successfully!', 'success')
        # Redirect to add_decision again with a fresh form
        return redirect(url_for('decision.add_decision'))
    
    return render_template('add_decision.html', form=form)

@decision_bp.route('/decisions')
def view_decisions():
    """View all decisions."""
    decisions = decision_mirror.get_decisions()
    return render_template('decisions.html', decisions=decisions)

@decision_bp.route('/decision/<int:decision_id>')
def view_decision(decision_id):
    """View a specific decision."""
    decision = decision_mirror.get_decision(decision_id)
    if not decision:
        flash('Decision not found', 'danger')
        return redirect(url_for('decision.view_decisions'))
    
    return render_template('decision_detail.html', decision=decision)

@decision_bp.route('/delete_decision/<int:decision_id>', methods=['POST'])
def delete_decision(decision_id):
    """Delete a decision."""
    success = decision_mirror.delete_decision(decision_id)
    if success:
        flash(f'Decision #{decision_id} deleted successfully', 'success')
    else:
        flash(f'Failed to delete decision #{decision_id}', 'danger')
    
    return redirect(url_for('decision.view_decisions'))