import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Steps from './components/Steps.jsx'
import DropdownSelector from './components/Dropdown.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <div className='p-12'>
      <App />
    </div>
  </StrictMode>
)