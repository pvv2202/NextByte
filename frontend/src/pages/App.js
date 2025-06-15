import '../App.css';
import { useState} from 'react';
// components
import Workspace from '../components/workspace';
import Sidebar from '../components/sidebar';
import SidebarBtn from '../components/sidebarBtn';
import Login from './Login';
import { Navigate, Route, Routes } from 'react-router';
import SignUp from './SignUp';

/* First functional component */ 
function App() {
  // most used react hook -> allows you to track and update any variable 
  const [sidebarHidden, setSidebarHidden] = useState(false)
  const [loggedIn, setLoggedIn] = useState(false)

  return (
    // im using tailwind-css extension to style quicker, these random looking strings correspond to css styles
    // ie w-screen = width: 100vw (div total width of screen), p-0 (0 padding between div and children) 
    <div className="app flex w-screen min-h-screen bg-green-200 p-0">
      <Routes>
        <Route path='/' element={<Navigate to='/signup' />} />
        <Route path='/login' element={<Login />} />
        <Route path='signup' element={<SignUp />} />
        <Route path='/workspace' element={
          <>
            <Sidebar 
              sidebarHidden={sidebarHidden}
              setSidebarHidden={setSidebarHidden}
            />
            <SidebarBtn 
              sidebarHidden={sidebarHidden}
              setSidebarHidden={setSidebarHidden}
            />
            <Workspace sidebarHidden={sidebarHidden}/>
          </>
        }/>
      </Routes> 
    </div>
  );
}

export default App;
