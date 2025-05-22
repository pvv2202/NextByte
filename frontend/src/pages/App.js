import '../App.css';
import { useState} from 'react';
// components
import Workspace from '../components/workspace';
import Sidebar from '../components/sidebar';
import SidebarBtn from '../components/sidebarBtn';

/* First functional component */ 
function App() {
  // most used react hook -> allows you to track and update any variable 
  const [sidebarHidden, setSidebarHidden] = useState(true)

  return (
    // im using tailwind-css extension to style quicker, these random looking strings correspond to css styles
    // ie w-screen = width: 100vw (div total width of screen), p-0 (0 padding between div and children) 
    <div className="flex w-screen min-h-screen bg-green-200 p-0">
      <SidebarBtn 
        // props -> let you pass js variables down to child components
        sidebarHidden={sidebarHidden}
        setSidebarHidden={setSidebarHidden}
      />
      <Sidebar sidebarHidden={sidebarHidden}/>
      <Workspace sidebarHidden={sidebarHidden}/>
      
    </div>
  );
}

export default App;
