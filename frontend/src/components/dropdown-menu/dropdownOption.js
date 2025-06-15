import React from 'react'
import DropdownItem from './dropdownItem'

function DropdownOptions({hideOptions, options, handleOptionClick, imgpath, valueKey}) {

  return (
    <ul className={`${hideOptions ? 'opacity-0 max-h-0 pointer-events-none' : 'opacity-100 max-h-40'} 
    absolute z-[100] top-12 bg-white w-[300px] max-h-60 overflow-auto transition-all ease-in-out duration-300 bg-white rounded-xl shadow-lg shadow-green-400`}>
        {options.map((option, idx) => (
          <DropdownItem
              handleOptionClick={handleOptionClick}
              imgpath={imgpath}
              valueKey={valueKey}
              option={option}
              key={idx}
              
            />   
        ))}
    </ul>
  )
}

export default DropdownOptions