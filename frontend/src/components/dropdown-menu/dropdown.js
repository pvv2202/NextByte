import React from 'react'
import { useState, useEffect, useRef } from 'react'
import { faCaretDown, faCaretUp } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import DropdownOptions from './dropdownOption'

function DropdownMenu({options, onChange, imgpath, category, valueKey}) {
    const [search, setSearch] = useState('')
    const [hideOptions, setHideOptions] = useState(true)
    const dropdownRef = useRef(null) // 1. Create a ref

    
    useEffect(() => {
        function handleClickOutside(event) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setHideOptions(true)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => {
            document.removeEventListener('mousedown', handleClickOutside)
        }
    }, [])
    
    const handleInput = (e) => {
        setHideOptions(false)
        setSearch(e.target.value)
        onChange('')
    }
    const handleOptionClick = (option) => {
        setHideOptions(true)
        setSearch(option[valueKey])
        onChange(option[valueKey])
    } 
    
    return (
        <div ref={dropdownRef} className='relative flex flex-col relative w-40'>
            <div className='flex items-center'>
                <input type="text" className='w-full p-2' placeholder={`Select ${category}`} value={search.length > 15? search.slice(0,15) + '...' : search} onChange={handleInput}/>
                <button className='absolute right-1' type='button' onClick={() => setHideOptions(h => !h)}>
                    {hideOptions ? <FontAwesomeIcon icon={faCaretDown} /> : <FontAwesomeIcon icon={faCaretUp} /> }
                    </button>
            </div>
            <DropdownOptions
                options={options.filter(option => option[valueKey].toLowerCase().includes(search.toLowerCase()))}
                hideOptions={hideOptions}
                setHideOptions={setHideOptions}
                handleOptionClick={handleOptionClick}
                imgpath={imgpath}
                valueKey={valueKey}
            />
        </div>
    )
}

export default DropdownMenu