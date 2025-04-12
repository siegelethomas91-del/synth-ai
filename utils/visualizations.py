def show_visualizations(data: pd.DataFrame, engine: DataEngine):
    """Display engine-specific visualizations"""
    if engine == DataEngine.FINANCE:
        # Finance visualizations
        fig_amount = px.histogram(data, x='amount', title='Transaction Amount Distribution')
        st.plotly_chart(fig_amount)
        
        fig_fraud = px.pie(data, names='is_fraud', title='Fraud Distribution')
        st.plotly_chart(fig_fraud)
    
    elif engine == DataEngine.HEALTHCARE:
        # Healthcare visualizations
        fig_age = px.histogram(data, x='patient_age', title='Patient Age Distribution')
        st.plotly_chart(fig_age)
        
        fig_diagnosis = px.pie(data, names='diagnosis', title='Diagnosis Distribution')
        st.plotly_chart(fig_diagnosis)
    
    elif engine == DataEngine.LLM:
        # LLM visualizations
        fig_length = px.histogram(data, x='prompt_length', title='Prompt Length Distribution')
        st.plotly_chart(fig_length)
        
        fig_sentiment = px.pie(data, names='sentiment', title='Sentiment Distribution')
        st.plotly_chart(fig_sentiment)