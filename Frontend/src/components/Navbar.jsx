import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';
import logo from '../assets/logo.png';

const Navbar = ({ darkMode, toggleDarkMode }) => {
  useEffect(() => {
    // Only initialize once
    if (window.__googleTranslateInitialized) return;

    window.__googleTranslateInitialized = true; // Mark as initialized

    const script = document.createElement('script');
    script.src = "//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";
    script.async = true;
    document.body.appendChild(script);

    window.googleTranslateElementInit = () => {
      new window.google.translate.TranslateElement({
        pageLanguage: 'en',
        includedLanguages: 'en,hi,ta,te,ml,kn,gu,bn,mr,pa,ur',
        layout: window.google.translate.TranslateElement.InlineLayout.SIMPLE
      }, 'google_translate_element');
    };
  }, []);

  return (
    <nav className={`navbar ${darkMode ? 'dark-mode' : ''}`}>
      <div className="logo-container">
        <img src={logo} alt="TrustReview Logo" className="logo-img" />
        <span className="logo-text">TrustReview</span>
      </div>

      <ul className="nav-links">
        <li><Link to="/">Home</Link></li>
        <li><Link to="/how-it-works">How it Works</Link></li>
        <li><Link to="/features">Features</Link></li>
        <li><Link to="/about">About</Link></li>
      </ul>

      <div className="nav-actions">
        {/* Google Translate Dropdown */}
        <div id="google_translate_element" className="translate-widget"></div>

        {/* Theme Toggle */}
        <button className="theme-toggle" onClick={toggleDarkMode}>
          {darkMode ? '☀️' : '🌙'}
        </button>

        {/* Login & Get Started */}
        <Link to="/login" className="login-btn">Login</Link>
        <Link to="/signup" className="get-started-btn">Get Started</Link>
      </div>
    </nav>
  );
};

export default Navbar;
