import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-grid">
        <div className="footer-brand">
          <h3>TrustReview</h3>
          <p>AI System To Review Authenticity</p>
        </div>

        <div>
          <h4>Product</h4>
          <ul>
            <li><Link to="/pricing">Pricing</Link></li>
            <li><Link to="/features">Features</Link></li>
          </ul>
        </div>

        <div>
          <h4>Company</h4>
          <ul>
            <li><Link to="/about">About</Link></li>
          </ul>
        </div>

        <div>
          <h4>Connect</h4>
          <div className="social-icons">
            <div className="social-item">
              <a 
                href="https://www.linkedin.com/in/ashok-j-93691b309/" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                🔗 LinkedIn
              </a>
            </div>
            <div className="social-item">
              <a href="mailto:ashokj.23cse@kongu.edu">
                📧 ashokj.23cse@kongu.edu
              </a>
            </div>
          </div>
        </div>
      </div>

      <div className="footer-bottom">
        <p>© 2025 TrustReview. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;
