import { faBook, faGear, faKitchenSet, faUserGroup, faSignOut} from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { Link, useNavigate} from 'react-router';
import { api_request } from '../api_request';
import Avatar from '@mui/material/Avatar'
import { useState, useEffect, useContext} from 'react';
import { UserContext } from '../UserContext';
import {motion, AnimatePresence} from 'framer-motion'



function Sidebar({sidebarHidden}) {
    const navigate = useNavigate()
    const {user, setUser} = useContext(UserContext)

    

    const handleSignout = async () => {
          
          try{
                // ensure server-side session is removed 
                const response = await api_request(
                    'auth/signout',
                    'POST'
                )
                // ensure user state is removed
                setUser(null)
                // go back to login page
                navigate('/login')
              
          } catch(err){
             console.log(err)
              
          }
              
    }

    if (!user) return <div>loading</div>
    return (
        <AnimatePresence>
            <motion.div
                className={`sidebar flex flex-col gap-y-10 text-white transition-transform duration-100 ease-in-out fixed top-0 left-0 h-screen w-40 p-2 bg-sky-950 ${sidebarHidden ?'-translate-x-full': 'translate-x-0'}`}
                initial={{ opacity: 0.5 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5}}
            >
                <div className='flex flex-col relative gap-y-4 w-full items-center '>
                    <h2 className='font-semibold text-lg text-center'>NextByte</h2>
                    <Avatar sx={{width: 55, height:55, bgcolor: 'green', color: 'whitesmoke'}}>{user['username'][0].toUpperCase()}</Avatar>      
                </div>
                <ul className='flex flex-col relative top-8 gap-y-8 w-full '>
                    <Link to='my-recipes'>
                        <li className='flex gap-x-2 items-center'>
                            <p>My Recipes</p>
                            <FontAwesomeIcon icon={faBook} />    
                        </li>
                    </Link>
                    <Link to='/landing'>
                        <li className='flex gap-x-2 items-center'>
                            <p>Create</p>
                            <FontAwesomeIcon icon={faKitchenSet} />   
                        </li>
                    </Link>
                    <li className='flex gap-x-2 items-center'>
                        <p>Kitchen Share-in progress</p>
                        <FontAwesomeIcon icon={faUserGroup} />
                    </li>
                    <li className='flex gap-x-2 items-center'>
                        <p>Account-in progress</p>
                        <FontAwesomeIcon icon={faGear} />
                    </li>
                    <li className=''>
                        <button className='flex gap-x-2 items-center' onClick={handleSignout}>
                            <p>Sign out</p>
                            <FontAwesomeIcon icon={faSignOut} />
                        </button>
                        
                    </li>
                </ul>
            </motion.div>
            
        </AnimatePresence>
    )
}

export default Sidebar