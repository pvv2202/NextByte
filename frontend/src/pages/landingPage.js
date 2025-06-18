
import Workspace from '../components/workspace';
import Sidebar from '../components/sidebar';
import SidebarBtn from '../components/sidebarBtn';
import { get_user } from '../api_request';
import { useNavigate } from 'react-router';
import { useState, useEffect, useContext} from 'react';
import { UserContext } from '../UserContext';



function LandingPage() {
    const [sidebarHidden, setSidebarHidden] = useState(false)
    const navigate = useNavigate()
    const {user, setUser} = useContext(UserContext)


    // when rendered, if the userState is null (not logged in) immediately navigate to login page
    useEffect(() => {
        if (!user) navigate('/login')
    }, [])
    
    
    return (
        <>
            <Sidebar 
                sidebarHidden={sidebarHidden}
                setSidebarHidden={setSidebarHidden}
            />
            <SidebarBtn 
                sidebarHidden={sidebarHidden}
                setSidebarHidden={setSidebarHidden}
            />
            <Workspace 
                sidebarHidden={sidebarHidden}
            />
    
        </>
    )
}
            

export default LandingPage