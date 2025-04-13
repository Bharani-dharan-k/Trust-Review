import React from 'react';
import './Pages.css';

const HowItWorks = () => {
  return (
    <section className="page-section">
      <h2>How It Works</h2>
      <p>
        Online reviews heavily influence consumer decisions — from buying gadgets to selecting restaurants. 
        However, a significant number of these reviews are fake, posted by bots or paid individuals with hidden agendas. 
        TrustReview aims to fix that using AI.
      </p>

      <p>
        Our AI system follows a multi-layered approach to detect fake reviews:
      </p>

      <ul>
        <li>
          <strong>Language Pattern Analysis:</strong> 
          We use Natural Language Processing (NLP) to identify repetitive phrases, exaggerated tones, and keyword stuffing common in fake reviews.
        </li>
        <li>
          <strong>User Review History:</strong> 
          The system examines the reviewer's past behavior, like sudden bursts of reviews, suspicious posting frequency, or promoting the same product type.
        </li>
        <li>
          <strong>Sentiment Analysis:</strong> 
          Emotionally charged or overly positive/negative reviews are analyzed for authenticity using sentiment scoring algorithms.
        </li>
        <li>
          <strong>Reviewer Credibility Score:</strong> 
          Each reviewer is assigned a trust score based on their review consistency, diversity, and interaction patterns.
        </li>
        <li>
          <strong>Visual Flags & Suggestions:</strong> 
          The end-user is shown confidence levels or warnings when a review is suspected to be fake.
        </li>
      </ul>

      <p>
        By combining machine learning models with real-time data monitoring, TrustReview empowers users to make smarter, evidence-backed choices online.
      </p>
    </section>
  );
};

export default HowItWorks;
