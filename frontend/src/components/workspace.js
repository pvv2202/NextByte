import React from 'react'
import CreatePage from '../pages/createPage'


function Workspace({sidebarHidden}) {
  return (
    <div className={`workspace flex justify-center h-full bg-gray-800 w-full transition-all duration-100 p-0 ${sidebarHidden ? 'ml-0' : 'ml-40'}`}>
        <CreatePage />   
    </div>
  )
}

export default Workspace