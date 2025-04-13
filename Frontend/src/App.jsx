import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './components/Home';
import HowItWorks from './components/HowItWorks';
import Features from './components/Features';
import About from './components/About';
import Footer from './components/Footer';
import Signup from './components/Signup';
import Login from './components/Login';
import Chatbot from './components/Chatbot';
import ReviewDetector from './components/ReviewDetector'; // 👈 ADD this import
import './App.css';
import ReviewChecker from './components/ReviewChecker';
import MultipleReview from './components/MultipleReview'; // 👈 ADD this import
function App() {
  const [darkMode, setDarkMode] = useState(false);

  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
  };

  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  return (
    <Router>
      <div className="app-container">
        <Navbar darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/how-it-works" element={<HowItWorks />} />
            <Route path="/features" element={<Features />} />
            <Route path="/about" element={<About />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="/login" element={<Login />} />
            <Route path="/review-detector" element={<ReviewDetector />} /> {/* 👈 NEW Route */}
            <Route path="/review-checker" element={<ReviewChecker />} /> {/* 👈 NEW Route */}
            <Route path="/multiple" element={<MultipleReview />} /> {/* 👈 NEW Route */}
          </Routes>
        </main>
        <Chatbot />
        <Footer />
      </div>
    </Router>
  );
}

export default App;
