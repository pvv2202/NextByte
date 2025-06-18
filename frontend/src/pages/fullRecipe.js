import React from 'react'
import { useParams, useNavigate } from 'react-router'
import { useState, useEffect } from 'react'
import { api_request } from '../api_request'
import {motion, AnimatePresence} from 'framer-motion'

function FullRecipe() {
  const navigate = useNavigate()
  const {recipe_id} = useParams();
  const [loading, setLoading] = useState(true)
  const [recipe, setRecipe] = useState(null)
 
 
  useEffect(() => {
    async function getRecipe () {
        setLoading(true)
        const recipe = await api_request(
          `recipes/saved-recipes/${recipe_id}`,
          'GET'
        )
        console.log(recipe)
     
        setRecipe(recipe)

        setLoading(false)
         
    }
    getRecipe()
  }, [])

  if(loading) return <div>loading...</div>




  return (
    <div className='relative flex flex-col w-full min-h-screen gap-y-20 items-center bg-white'>
        <AnimatePresence>
          <motion.div 
            className='w-full h-full flex flex-col bg-white p-8 rounded-xl shadow-md'
         
            exit={{ scale: 0 }}
              
          >
            <div className='flex gap-1 justify-between'>
              <h2 className='text-2xl text-center '>
                {recipe[1].split(" ").map((el, i) => (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{
                      duration: 0.5,
                      delay: i / 10
                    }}
                    key={i}
                    >
                    {el}{" "}
                    </motion.span>
                ))}
              </h2>  
            </div> 
            <div className='w-full flex p-2'>
              <div className='w-40 object-contain border-2'>
                  {!recipe[4] ? <img src='../../nextbyte.png'/> : ''}
              </div>
            </div>

            <ul className='ml-4 mt-2 list-disc'>
              {recipe[2].split("||").map((ing, idx)=>(
                <motion.li key={idx}>
                  {ing.split(" ").map((el, i) => (
                    <motion.span
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{
                        duration: 0.5,
                        delay: i / 10
                      }}
                      key={i}
                      >
                      {el}{" "}
                      </motion.span>
                ))}
                </motion.li> ))}
              </ul>


            <ol className='ml-4 mt-2 list-decimal'>
                {recipe[3].split("||").map((dir, idx)=>(
                  <motion.li key={idx}>
                    {dir.split(" ").map((el, i) => (
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{
                          duration: 0.5,
                          delay: i / 10
                        }}
                        key={i}
                        >
                        {el}{" "}
                        </motion.span>
                    ))}
                  </motion.li>))
                  }
              </ol>
              <div className='w-full flex justify-end'>
                Created: {recipe[5]}
              </div>

        </motion.div>
      </AnimatePresence>
    </div>
  )
}

export default FullRecipe