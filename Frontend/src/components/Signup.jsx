import React, { useState } from 'react';
import './Signup.css';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Signup = () => {
  const navigate = useNavigate(); // Initialize navigation hook

  const [form, setForm] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: ''
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.id]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.post('http://localhost:5000/api/auth/signup', form);
      alert("Signup successful!");
      navigate('/login'); // Redirect to login page
    } catch (err) {
      alert(err.response?.data?.message || "Signup failed");
    }
  };

  return (
    <div className="signup-wrapper">
      <div className="auth-card active">
        <h1>Create Your Account</h1>
        <p className="subtitle">Enterprise-Grade Review Verification System</p>

        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <div>
              <label htmlFor="firstName">First Name</label>
              <div className="input-field">
                <input
                  type="text"
                  id="firstName"
                  placeholder="First Name"
                  onChange={handleChange}
                  required
                />
              </div>
            </div>
            <div>
              <label htmlFor="lastName">Last Name</label>
              <div className="input-field">
                <input
                  type="text"
                  id="lastName"
                  placeholder="Last Name"
                  onChange={handleChange}
                  required
                />
              </div>
            </div>
          </div>

          <label htmlFor="email">Email</label>
          <div className="input-field">
            <input
              type="email"
              id="email"
              placeholder="Email"
              onChange={handleChange}
              required
            />
          </div>

          <label htmlFor="password">Password</label>
          <div className="input-field">
            <input
              type="password"
              id="password"
              placeholder="Password"
              onChange={handleChange}
              required
            />
          </div>

          <button className="auth-btn" type="submit">Sign Up</button>
        </form>
      </div>
    </div>
  );
};

export default Signup;
