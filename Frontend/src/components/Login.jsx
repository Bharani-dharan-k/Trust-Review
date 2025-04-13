import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom'; // 👈 Import this
import './Login.css';

const Login = () => {
  const [form, setForm] = useState({ email: '', password: '' });
  const navigate = useNavigate(); // 👈 Initialize

  const handleChange = (e) => {
    setForm({ ...form, [e.target.id]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:5000/api/auth/login', form);
      alert("Login successful!");
      localStorage.setItem('token', res.data.token);
      navigate('/review-checker'); // 👈 Navigate after login
    } catch (err) {
      alert(err.response?.data?.message || "Login failed");
    }
  };

  return (
    <div className="login-page">
      <div className="auth-card">
        <h1>Login</h1>
        <form onSubmit={handleSubmit}>
          <label htmlFor="email">Email</label>
          <div className="input-field">
            <input type="email" id="email" placeholder="Enter your email" onChange={handleChange} required />
          </div>

          <label htmlFor="password">Password</label>
          <div className="input-field">
            <input type="password" id="password" placeholder="Enter your password" onChange={handleChange} required />
          </div>

          <button className="auth-btn" type="submit">Login</button>
        </form>
      </div>
    </div>
  );
};

export default Login;