import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './pages/App';
import {BrowserRouter} from "react-router"

/* 
React provides a really great format for structuring, displaying and updating web elements
It works by creating reusable functional components, see ./components that return html objects which are
then rendered on the document object model (DOM) of the webpage. The dom is a tree structure that allows
for easy access to any element. Think nodes -> parent nodes have child nodes in a tree-> in the DOM parent 
elements (like a container) have child elements inside them. the dom captures these relationships
*/
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter> 
  </React.StrictMode>
);


