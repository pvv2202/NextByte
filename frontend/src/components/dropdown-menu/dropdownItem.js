import React from 'react'

function DropdownItem({handleOptionClick, option, valueKey, imgpath}) {

    const getimgpath = (start, keys) => {
        let imgpath = start
        for(const key of keys){
        imgpath = imgpath[key]
        }
        return imgpath
    }

    return (
            <li className='flex justify-between w-full p-2
            hover:bg-sky-200 cursor-pointer items-center' onClick={() => handleOptionClick(option)}>
                {option[valueKey]}
                {imgpath.length < 1 ? '' : <img src={`${getimgpath(option, imgpath)}`} className='w-8 h-8 object-contain' alt="" />}   
            </li>
    )
}

export default DropdownItem