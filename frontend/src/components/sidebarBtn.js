import React from 'react'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {faCoffee, faWindowMaximize} from '@fortawesome/free-solid-svg-icons';

function SidebarBtn({sidebarHidden, setSidebarHidden, }) {
  
    return (
        <button
        className={`fixed hover:scale-105 top-0 z-50 bg-gray-800 text-white rounded p-2 transition-all duration-100 ${sidebarHidden ? '-left-1' : 'left-36'}`}
        onClick={() => setSidebarHidden(h => !h)}
        >
            <FontAwesomeIcon icon={faWindowMaximize} />
        </button>
        
    )
}


export default SidebarBtn