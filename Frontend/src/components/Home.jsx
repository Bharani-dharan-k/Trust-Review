import React from 'react';
import './Home.css';
import robotImage from '../assets/robot.png';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  const handleTryNow = () => {
    navigate('/signup');
  };

  const handleLearnMore = () => {
    navigate('/how-it-works');
  };

  return (
    <div className="home">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-text">
          <h1>Unmask Fake Reviews with AI-Powered Detection</h1>
          <p>
            Make informed decisions with our advanced AI system that detects suspicious activity using language patterns, user history, and sentiment analysis.
          </p>
          <div className="hero-buttons">
            <button className="btn-blue" onClick={handleTryNow}>Try It Now</button>
            <button className="btn-outline" onClick={handleLearnMore}>Learn More</button>
          </div>
        </div>
        <div className="hero-img">
          <img src={robotImage} alt="AI Robot" />
        </div>
      </section>

      {/* Detection Section */}
      <section className="detect">
        <h2>How We Detect Fake Reviews</h2>
        <div className="cards">
          <div className="card">
            <h4>Language Pattern Analysis</h4>
            <p>AI algorithms analyze suspicious tone and keywords.</p>
          </div>
          <div className="card">
            <h4>User History Tracking</h4>
            <p>Track user behavior to find patterns.</p>
          </div>
          <div className="card">
            <h4>Sentiment Analysis</h4>
            <p>Evaluate emotions and authenticity.</p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
